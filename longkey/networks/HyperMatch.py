import math
import torch
import logging
from torch import nn
from torch.nn import MarginRankingLoss
import geoopt as gt
from .hyperbolic.poincare import PoincareBall
from .hyperbolic.mobius_linear import MobiusLinear
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel

logger = logging.getLogger()


class NGramers(nn.Module):
    def __init__(self, input_size, hidden_size, max_gram, dropout_rate):
        super().__init__()

        self.cnn_list = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=input_size, out_channels=hidden_size, kernel_size=n
                )
                for n in range(1, max_gram + 1)
            ]
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.transpose(1, 2)

        outputs = None
        for cnn in self.cnn_list:
            y = cnn(x)
            y = self.relu(y)
            y = self.dropout(y)
            outputs = (
                y.transpose(1, 2)
                if outputs is None
                else torch.cat([outputs, y.transpose(1, 2)], dim=1)
            )

        return outputs


# -------------------------------------------------------------------------------------------
# Inherit PreTrainedModel
# -------------------------------------------------------------------------------------------
class HyperMatch(PreTrainedModel):
    def __init__(self, config):
        super(HyperMatch, self).__init__(config)

        max_gram = config.max_phrase_words
        cnn_dropout_rate = config.hidden_dropout_prob

        self.use_checkpoint = config.use_checkpoint
        self.max_encoder_token_size = config.max_encoder_token_size
        self.num_labels = config.num_labels
        self.encoder = None

        self.cnn2gram = NGramers(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            max_gram=max_gram,
            dropout_rate=cnn_dropout_rate,
        )

        self.classifier = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.sigmoid = nn.Sigmoid()
        self.scalar_layer = AdaptiveMixingLayer(hidden_size=config.hidden_size)
        self.rank_value = config.hidden_size

        self.hyper_rank = nn.Parameter(
            torch.Tensor(config.hidden_size, config.hidden_size)
        )
        nn.init.kaiming_normal_(self.hyper_rank, mode="fan_in", nonlinearity="relu")
        self.p_ball = gt.PoincareBall()
        self.c = 1
        self.hyper_linear = MobiusLinear(
            manifold=PoincareBall(),
            in_features=config.hidden_size,
            out_features=1,
            c=self.c,
        )
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        global_attention_mask,
        valid_ids,
        active_mask,
        valid_output,
        chunk_mask=None,
        labels=None,
    ):
        """
        active_mask : mention_mask for ngrams = torch.LongTensor([[1,2,1,3,4,5,4], [1,2,3,0,4,4,0]])
        labels : for ngrams labels = torch.LongTensor([[1,-1,-1,1,-1], [1,-1,-1,1,0]])
        """
        # --------------------------------------------------------------------------------
        # Encoder Embedding Outputs

        device = input_ids.get_device()

        max_token_size = input_ids.size()[1]
        block_size = math.ceil(
            max_token_size / math.ceil(max_token_size / self.max_encoder_token_size)
        )
        input_ids = [x for x in torch.split(input_ids, block_size, dim=1)]
        attention_mask = [x for x in torch.split(attention_mask, block_size, dim=1)]
        if global_attention_mask is not None:
            global_attention_mask = [
                x for x in torch.split(global_attention_mask, block_size, dim=1)
            ]

        sequence_output = list()

        for i in range(len(input_ids)):
            if self.training and self.use_checkpoint:
                encoder_args = [self.encoder, input_ids[i], attention_mask[i]]

                if global_attention_mask is not None:
                    encoder_args += [global_attention_mask[i]]

                output = checkpoint(*encoder_args, use_reentrant=False)
            else:
                encoder_args = {
                    "input_ids": input_ids[i],
                    "attention_mask": attention_mask[i],
                }
                if global_attention_mask is not None:
                    encoder_args["global_attention_mask"] = global_attention_mask[i]

                output = self.encoder(**encoder_args)

            outputs_final = torch.stack(output[2])[1:13]

            sequence_output += [self.scalar_layer(outputs_final)]

        sequence_output = torch.cat(sequence_output, axis=1)

        # --------------------------------------------------------------------------------
        # Valid Outputs : get first token vector
        batch_size = sequence_output.size(0)
        valid_output = valid_output.type(sequence_output.dtype)

        for i in range(batch_size):
            vectors = sequence_output[i][valid_ids[i] == 1]
            valid_output[i, : sum(valid_ids[i]).item()].copy_(vectors)

        # --------------------------------------------------------------------------------
        # Dropout
        sequence_output = self.dropout(valid_output)

        hyper_sequence_output = self.p_ball.expmap0(
            torch.matmul(sequence_output, self.hyper_rank)
        )
        mean_document = self.einstein_midpoint(hyper_sequence_output).unsqueeze(1)

        # --------------------------------------------------------------------------------
        # CNN Outputs
        if self.training and self.use_checkpoint:
            cnn_outputs = checkpoint(
                self.cnn2gram, sequence_output, use_reentrant=False
            )
        else:
            cnn_outputs = self.cnn2gram(
                sequence_output
            )  # shape = (batch_size, max_gram_num, cnn_output_size)

        ngrams_outputs = self.p_ball.expmap0(cnn_outputs)
        # --------------------------------------------------------------------------------
        # Importance Scores
        classifier_scores1 = -self.p_ball.dist(
            ngrams_outputs, mean_document.expand_as(ngrams_outputs)
        ) / math.sqrt(self.rank_value)
        classifier_scores2 = self.hyper_linear(
            ngrams_outputs.view(-1, self.rank_value)
        ).view(batch_size, -1)
        classifier_scores = 0.5 * classifier_scores1 + 0.5 * classifier_scores2

        classifier_scores = classifier_scores.squeeze(
            -1
        )  # shape = (batch_size, max_gram_num)

        total_scores = list()
        total_scores_temp = active_mask.type(classifier_scores.dtype)
        for i in range(batch_size):
            valid_indexes = chunk_mask[i] != -1
            chunk_mask_ids = chunk_mask[i][valid_indexes]

            cur_total_scores = total_scores_temp[i].scatter_reduce(
                0,
                chunk_mask_ids,
                classifier_scores[i][valid_indexes],
                reduce="amax",
                include_self=False,
            )

            total_scores += [cur_total_scores]

        total_scores = torch.stack(total_scores, dim=0)

        if len(total_scores.shape) == 1:
            total_scores = total_scores.unsqueeze(0)

        # --------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------
        # Total Loss Compute
        if labels is not None:
            Rank_Loss_Fct = MarginRankingLoss(
                margin=1.0 / math.sqrt(self.rank_value), reduction="mean"
            )
            flag = torch.FloatTensor([1]).to(device)

            rank_losses = None
            for i in range(batch_size):
                score = total_scores[i]
                label = labels[i]

                true_score = score[label == 1]
                neg_score = score[label == -1]

                rank_loss = Rank_Loss_Fct(
                    true_score.unsqueeze(-1), neg_score.unsqueeze(0), flag.unsqueeze(0)
                ).unsqueeze(0)
                rank_losses = (
                    rank_loss
                    if rank_losses is None
                    else torch.cat([rank_losses, rank_loss], dim=0)
                )

            rank_loss = torch.mean(rank_losses)
            return rank_loss

        else:
            return total_scores  # shape = (batch_size * max_differ_gram_num)

    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c**0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def klein_constraint(self, x):
        last_dim_val = x.size(-1)
        norm = torch.reshape(torch.norm(x, dim=-1), [-1, 1])
        maxnorm = 1 - self.eps[x.dtype]
        cond = norm > maxnorm
        x_reshape = torch.reshape(x, [-1, last_dim_val])
        projected = x_reshape / (norm + self.min_norm) * maxnorm
        x_reshape = torch.where(cond, projected, x_reshape)
        x = torch.reshape(x_reshape, list(x.size()))
        return x

    def to_klein(self, x, c=1):
        x_2 = torch.sum(x * x, dim=-1, keepdim=True)
        x_klein = 2 * x / (1.0 + x_2)
        x_klein = self.klein_constraint(x_klein)
        return x_klein

    def klein_to_poincare(self, x, c=1):
        x_poincare = x / (
            1.0 + torch.sqrt(1.0 - torch.sum(x * x, dim=-1, keepdim=True))
        )
        x_poincare = self.proj(x_poincare, c)
        return x_poincare

    def lorentz_factors(self, x):
        x_norm = torch.norm(x, dim=-1)
        return 1.0 / (1.0 - x_norm**2 + self.min_norm)

    def einstein_midpoint(self, x, c=1):
        x = self.to_klein(x, c)
        x_lorentz = self.lorentz_factors(x)
        x_norm = torch.norm(x, dim=-1)
        # deal with pad value
        x_lorentz = (1.0 - torch._cast_Float(x_norm == 0.0)) * x_lorentz
        x_lorentz_sum = torch.sum(x_lorentz, dim=-1, keepdim=True)
        x_lorentz_expand = torch.unsqueeze(x_lorentz, dim=-1)
        x_midpoint = torch.sum(x_lorentz_expand * x, dim=1) / x_lorentz_sum
        x_midpoint = self.klein_constraint(x_midpoint)
        x_p = self.klein_to_poincare(x_midpoint, c)
        return x_p


class AdaptiveMixingLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.v_q = nn.Parameter(torch.Tensor(hidden_size, 1))
        nn.init.kaiming_normal_(self.v_q, mode="fan_in", nonlinearity="relu")

        self.W_q = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        nn.init.kaiming_normal_(self.W_q, mode="fan_in", nonlinearity="relu")

    def forward(self, layer_representations):
        atten = torch.softmax(
            torch.matmul(layer_representations, self.v_q).permute(1, 2, 0, 3)
            / math.sqrt(self.hidden_size),
            2,
        )
        atten_h = torch.matmul(
            layer_representations.permute(1, 2, 3, 0), atten
        ).squeeze(-1)
        outputs = torch.matmul(atten_h, self.W_q)

        return outputs
