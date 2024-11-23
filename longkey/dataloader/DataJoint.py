import logging
import numpy as np
import random
import torch
from longkey import utils
import math
from .AbstractJoint import AbstractJoint

logger = logging.getLogger()


class DataJoint(AbstractJoint):
    def get_example(self, index, check_others=True, eval=False):
        cur_example = self.get_arrow_row_by_id(self.table, index)

        (
            url,
            tokens,
            valid_mask,
            mention_lists,
            ngram_label,
            chunk_label,
            phrase_list,
        ) = self._preprocessor(
            ex=cur_example,
            tokenizer=self.tokenizer,
            max_token=self.max_token,
            mode=self.mode,
            max_phrase_words=self.cfg.model.max_phrase_words,
            stem_flag=self.cfg.evaluate.stem,
        )

        while check_others and url is None:
            (
                url,
                tokens,
                valid_mask,
                mention_lists,
                ngram_label,
                chunk_label,
                phrase_list,
            ) = self.get_example(
                random.randint(0, self.__len__() - 1), check_others=False
            )

        if eval:
            return url, phrase_list
        else:
            return (
                url,
                tokens,
                valid_mask,
                mention_lists,
                ngram_label,
                chunk_label,
                phrase_list,
            )

    @classmethod
    def _preprocessor(
        cls, ex, tokenizer, max_token, mode, max_phrase_words, stem_flag=False
    ):
        tokens = valid_mask = mention_lists = ngram_label = chunk_label = (
            phrase_list
        ) = keyphrases = start_end_pos = None

        overlen_num = 0

        # tokenize
        (
            tokens,
            valid_mask,
            tok_to_orig_index,
            _,
            _,
        ) = utils.loader.tokenize(doc_words=ex["doc_words"], tokenizer=tokenizer)

        max_word = (
            max_token
            if max_token is None or len(tokens) < max_token
            else tok_to_orig_index[max_token - 1] + 1
        )

        url = ex["url"]
        tokens = tokens[:max_token]
        valid_mask = valid_mask[:max_token]
        doc_words = ex["doc_words"][:max_word]

        assert len(tokens) == len(valid_mask)

        # ---------------------------------------------------------------------------
        if mode == "train":
            keyphrases = ex["keyphrases"]
            start_end_pos = ex["start_end_pos"]
        # ---------------------------------------------------------------------------
        # obtain gram info and label
        (
            phrase_list,
            mention_lists,
            overlen_flag,
            ngram_label,
            chunk_label,
        ) = cls.get_ngram_info_label(
            doc_words, max_phrase_words, stem_flag, keyphrases, start_end_pos
        )

        if overlen_flag:
            overlen_num += 1
        # ---------------------------------------------------------------------------
        if mode == "train":
            if ngram_label is None:
                url = None
        # ---------------------------------------------------------------------------

        return (
            url,
            tokens,
            valid_mask,
            mention_lists,
            ngram_label,
            chunk_label,
            phrase_list,
        )

    @staticmethod
    def _converter(
        tokens,
        valid_mask,
        ngram_label,
        phrase_list,
        tokenizer,
        mode,
    ):
        """convert each batch data to tensor ; add [CLS] [SEP] tokens ;"""

        BOS_TOKEN = tokenizer.bos_token if tokenizer.bos_token else "[CLS]"
        src_tokens = np.append([BOS_TOKEN], tokens)
        valid_ids = np.append([0], valid_mask)

        src_tensor = torch.LongTensor(tokenizer.convert_tokens_to_ids(src_tokens))
        valid_mask = torch.LongTensor(valid_ids)

        orig_doc_len = sum(valid_ids)

        label = tot_phrase_len = None

        if mode == "train":
            label = torch.LongTensor(ngram_label)
        else:
            tot_phrase_len = len(phrase_list)

        return (
            src_tensor,
            valid_mask,
            orig_doc_len,
            label,
            tot_phrase_len,
        )

    @staticmethod
    def batchify_train(batch, encoder_output_dim, global_attention):
        """train dataloader & eval dataloader ."""

        ids = [ex[0] for ex in batch]
        docs = [ex[1] for ex in batch]
        valid_mask = [ex[2] for ex in batch]
        mention_mask = [ex[3] for ex in batch]
        doc_word_lens = [ex[4] for ex in batch]
        max_phrase_words = [ex[5] for ex in batch][0]

        # label
        label_list = [ex[6] for ex in batch]  # different ngrams numbers
        chunk_list = [ex[7] for ex in batch]  # whether is a chunk phrase

        max_word_len = max([word_len for word_len in doc_word_lens])  # word-level

        # ---------------------------------------------------------------
        # [1] [2] src tokens tensor
        doc_max_length = max([d.size(0) for d in docs])
        input_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
        input_mask = torch.LongTensor(len(docs), doc_max_length).zero_()
        global_attention_mask = (
            torch.LongTensor(len(docs), doc_max_length).zero_()
            if global_attention
            else None
        )

        for i, d in enumerate(docs):
            input_size = min(d.size(0), doc_max_length)
            input_ids[i, :input_size].copy_(d[:input_size])
            input_mask[i, :input_size].fill_(1)
            if global_attention_mask is not None:
                global_attention_mask[i, 0] = 1

        # ---------------------------------------------------------------
        # [3] valid mask tensor
        valid_max_length = max([v.size(0) for v in valid_mask])
        valid_ids = torch.LongTensor(len(valid_mask), valid_max_length).zero_()
        for i, v in enumerate(valid_mask):
            valid_ids_size = min(v.size(0), valid_max_length)
            valid_ids[i, :valid_ids_size].copy_(v[:valid_ids_size])

        # ---------------------------------------------------------------
        # [4] active mention mask : for n-gram (original)

        max_ngram_length = sum([max_word_len - n for n in range(max_phrase_words)])
        chunk_mask = torch.LongTensor(len(docs), max_ngram_length).fill_(-1)

        for batch_i, word_len in enumerate(doc_word_lens):
            pad_len = max_word_len - word_len

            batch_mask = []
            for n in range(max_phrase_words):
                ngram_len = word_len - n
                if ngram_len > 0:
                    assert len(mention_mask[batch_i][n]) == ngram_len
                    padding_array = np.array([-1 for _ in range(pad_len)])
                    gram_list = mention_mask[batch_i][n]

                    if len(padding_array) > 0:  # -1 for padding
                        gram_list = np.append(gram_list, padding_array)
                else:
                    gram_list = [-1 for _ in range(max_word_len - n)]
                batch_mask.extend(gram_list)

            if batch_mask:
                try:
                    chunk_mask[batch_i].copy_(torch.LongTensor(batch_mask))
                except RuntimeError as e:
                    raise RuntimeError

        # ---------------------------------------------------------------
        # [4] active mask : for n-gram
        max_diff_gram_num = 1 + max(
            [
                max(_mention_mask[len(_mention_mask) - 1])
                for _mention_mask in mention_mask
            ]
        )

        batch_size = len(docs)
        active_mask = torch.FloatTensor(batch_size, max_diff_gram_num).fill_(0)

        # -------------------------------------------------------------------
        # [5] label : for n-gram
        max_diff_grams_num = max([label.size(0) for label in label_list])
        ngram_label = torch.LongTensor(len(label_list), max_diff_grams_num).zero_()
        for batch_i, label in enumerate(label_list):
            ngram_label[batch_i, : label.size(0)].copy_(label)

        # -------------------------------------------------------------------
        # [6] Empty Tensor : word-level max_len
        valid_output = torch.zeros(len(docs), max_word_len, encoder_output_dim)

        # -------------------------------------------------------------------
        # [7] Chunk Label :
        max_chunks_num = max([chunks.size(0) for chunks in chunk_list])
        chunk_label = torch.LongTensor(len(chunk_list), max_chunks_num).fill_(-1)
        for batch_i, chunks in enumerate(chunk_list):
            chunk_label[batch_i, : chunks.size(0)].copy_(chunks)

        return (
            input_ids,
            input_mask,
            global_attention_mask,
            valid_ids,
            active_mask,
            valid_output,
            chunk_mask,
            ngram_label,
            chunk_label,
            ids,
        )

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
            "chunk_labels": batch[8],
        }

        return inputs, ex_indices
