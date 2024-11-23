import math
import torch
import logging
from torch import nn
from torch.nn import MarginRankingLoss, CrossEntropyLoss
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
class JointKPE(PreTrainedModel):
    def __init__(self, config):
        super(JointKPE, self).__init__(config)

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
        self.chunk_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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
        chunk_labels=None,
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

            sequence_output += [
                output[0],
            ]

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

        # --------------------------------------------------------------------------------
        # Classifier 512 to 1
        classifier_scores = self.classifier(
            cnn_outputs
        )  # shape = (batch_size, max_gram_num, 1)

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
        if labels is not None and chunk_labels is not None:
            # *************************************************************************************
            # *************************************************************************************
            # [1] Chunk Loss
            Chunk_Loss_Fct = CrossEntropyLoss(reduction="mean")

            active_chunk_loss = chunk_mask.view(-1) != -1
            chunk_logits = self.chunk_classifier(
                cnn_outputs
            )  # shape = (batch_size * num_gram, 2)
            active_chunk_logits = chunk_logits.view(-1, self.num_labels)[
                active_chunk_loss
            ]

            active_chunk_label_loss = chunk_labels.view(-1) != -1
            active_chunk_labels = chunk_labels.view(-1)[active_chunk_label_loss]

            chunk_loss = Chunk_Loss_Fct(active_chunk_logits, active_chunk_labels)

            # *************************************************************************************
            # *************************************************************************************
            # [2] Rank Loss
            Rank_Loss_Fct = MarginRankingLoss(margin=1, reduction="mean")
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
            # *************************************************************************************
            # *************************************************************************************
            # [3] Total Loss
            tot_loss = rank_loss + chunk_loss
            return tot_loss

        else:
            return total_scores  # shape = (batch_size * max_differ_gram_num)
