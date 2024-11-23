import math
import torch
import logging
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# CnnGram Extractor
# -------------------------------------------------------------------------------------------
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

        cnn_outpus = []
        for cnn in self.cnn_list:
            y = cnn(x)
            y = self.relu(y)
            y = self.dropout(y)
            cnn_outpus.append(y.transpose(1, 2))
        outputs = torch.cat(cnn_outpus, dim=1)
        return outputs


# -------------------------------------------------------------------------------------------
# Inherit PreTrainedModel
# -------------------------------------------------------------------------------------------
class ChunkKPE(PreTrainedModel):
    def __init__(self, config):
        super(ChunkKPE, self).__init__(config)

        max_gram = config.max_phrase_words
        cnn_dropout_rate = config.hidden_dropout_prob / 2

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

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
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
        labels=None,
    ):
        # --------------------------------------------------------------------------------
        # Encoder Embedding Outputs

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
        cnn_outputs = self.cnn2gram(
            sequence_output
        )  # shape = (batch_size, max_gram_num, cnn_output_size)

        logits = self.classifier(cnn_outputs)

        # activate loss
        active_ids = (
            active_mask.view(-1) == 1
        )  # [False, True, ...] # batch_size * max_len
        active_logits = logits.view(-1, self.num_labels)[active_ids]  # False

        if labels is not None:
            active_labels = labels.view(-1)[active_ids]
            loss_fct = CrossEntropyLoss(reduction="mean")
            loss = loss_fct(active_logits, active_labels)

            return loss
        else:
            return active_logits[:, 1]
