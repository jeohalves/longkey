import logging
from torch.utils.data import Dataset
from os.path import join
from tqdm import tqdm
import torch
from longkey import utils
from datasets import load_dataset

logger = logging.getLogger()


class AbstractData(Dataset):
    """build datasets for train & eval"""

    def __init__(self, cfg, tokenizer, max_token, mode, column_name="url"):
        self.mode = mode
        self.column_name = column_name
        self.cfg = cfg
        self.max_token = max_token
        self.tokenizer = tokenizer
        self.model_class = cfg.model.method
        self.max_phrase_words = cfg.model.max_phrase_words

        self.samples = load_dataset(
            "json",
            data_files=join(cfg.dir.data, cfg.data.dataset, f"{mode}.json"),
            split="train",
        )
        self.table = None

    def __len__(self):
        return len(self.samples)

    def get_arrow_row_by_id(self, table, row_id):
        return self.samples[row_id]

    def get_example(self, index, check_others=True, eval=False):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    @classmethod
    def _preprocessor():
        raise NotImplementedError()

    @staticmethod
    def _converter():
        raise NotImplementedError()

    @staticmethod
    def batchify_train():
        raise NotImplementedError()

    @classmethod
    def decoder(cls, cfg, data_loader, dataset, model, test_input_refactor, mode):
        logging.info("Start Generating Keyphrases for %s ... \n" % mode)

        candidate = dict()
        tot_predictions = list()

        for step, batch in enumerate(tqdm(data_loader, disable=cfg.rank != 0)):
            inputs, indices, lengths, phrase_list, url = test_input_refactor(
                batch, torch.device(model.cfg.device)
            )

            match cfg.mixed:
                case "fp16":
                    float_type = torch.float16
                case "bf16":
                    float_type = torch.bfloat16
                case _:
                    float_type = None

            if float_type:
                with torch.autocast(device_type="cuda", dtype=float_type):
                    logit_lists = cls.test(
                        model, inputs, lengths, cfg.model.max_phrase_words
                    )
            else:
                logit_lists = cls.test(
                    model, inputs, lengths, cfg.model.max_phrase_words
                )

            # decode logits to phrase per batch

            batch_predictions = cls.phrase(
                cfg=cfg,
                phrase_list=phrase_list,
                url=url,
                logit_lists=logit_lists,
                return_num=100,
            )

            tot_predictions.extend(batch_predictions)

        utils.pred_arranger(candidate, tot_predictions)
        return candidate

    @staticmethod
    def batchify_test():
        raise NotImplementedError()

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
            "labels": batch[6],
        }
        return inputs, ex_indices

    @staticmethod
    def test_input_refactor(batch, device):
        phrase_list, url, ex_indices, ex_phrase_numbers = (
            batch[-1],
            batch[-2],
            batch[-3],
            batch[-4],
        )
        batch = tuple(b.to(device) if b is not None else b for b in batch[:-4])
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "global_attention_mask": batch[2],
            "valid_ids": batch[3],
            "active_mask": batch[4],
            "valid_output": batch[5],
        }

        return inputs, ex_indices, ex_phrase_numbers, phrase_list, url

    @staticmethod
    def test():
        raise NotImplementedError()

    @classmethod
    def phrase():
        raise NotImplementedError()
