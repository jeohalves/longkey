import logging
from .AbstractRank import AbstractRank

logger = logging.getLogger()


class DataRank(AbstractRank):
    @staticmethod
    def train_input_refactor(batch, device):
        ex_indices = batch[-1]
        batch = tuple(b.to(device) if b is not None else b for b in batch[:-1])

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "global_attention_mask": batch[2],
            "valid_ids": batch[3],
            "active_mask": batch[4],
            "valid_output": batch[5],
            "chunk_mask": batch[6],
            "labels": batch[7],
        }

        return inputs, ex_indices
