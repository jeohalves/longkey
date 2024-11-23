import logging
import numpy as np
import random
import torch
from longkey import utils
import math
from longkey.utils.generator import remove_empty_phase, del_stemming_duplicate_phrase
from .AbstractData import AbstractData

logger = logging.getLogger()


class AbstractRank(AbstractData):
    def get_example(self, index, check_others=True, eval=False):
        cur_example = self.get_arrow_row_by_id(self.table, index)

        (
            url,
            tokens,
            valid_mask,
            mention_lists,
            ngram_label,
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
                phrase_list,
            )

    def __getitem__(self, index):
        (
            url,
            tokens,
            valid_mask,
            mention_lists,
            ngram_label,
            phrase_list,
        ) = self.get_example(index)

        (
            src_tensor,
            valid_mask,
            orig_doc_len,
            label,
            tot_phrase_len,
        ) = self._converter(
            tokens,
            valid_mask,
            ngram_label,
            phrase_list,
            self.tokenizer,
            self.mode,
        )

        if self.mode == "train":
            return (
                index,
                src_tensor,
                valid_mask,
                mention_lists,
                orig_doc_len,
                self.max_phrase_words,
                label,
                url,
                phrase_list,
            )
        else:
            return (
                index,
                src_tensor,
                valid_mask,
                mention_lists,
                orig_doc_len,
                self.max_phrase_words,
                tot_phrase_len,
                url,
                phrase_list,
            )

    @classmethod
    def _preprocessor(
        cls, ex, tokenizer, max_token, mode, max_phrase_words, stem_flag=False
    ):
        tokens = valid_mask = mention_lists = ngram_label = phrase_list = keyphrases = (
            start_end_pos
        ) = None

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

        return (
            input_ids,
            input_mask,
            global_attention_mask,
            valid_ids,
            active_mask,
            valid_output,
            chunk_mask,
            ngram_label,
            ids,
        )

    @staticmethod
    def batchify_test(batch, encoder_output_dim, global_attention):
        """test dataloader for Dev & Public_Valid."""

        ids = [ex[0] for ex in batch]
        docs = [ex[1] for ex in batch]
        valid_mask = [ex[2] for ex in batch]
        mention_mask = [ex[3] for ex in batch]
        doc_word_lens = [ex[4] for ex in batch]
        max_phrase_words = [ex[5] for ex in batch][0]

        phrase_list_lens = [ex[6] for ex in batch]
        url = [ex[7] for ex in batch]
        phrase_list = [ex[8] for ex in batch]

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
                chunk_mask[batch_i].copy_(torch.LongTensor(batch_mask))

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
        # [5] Empty Tensor : word-level max_len
        valid_output = torch.zeros(len(docs), max_word_len, encoder_output_dim)
        return (
            input_ids,
            input_mask,
            global_attention_mask,
            valid_ids,
            active_mask,
            valid_output,
            chunk_mask,
            phrase_list_lens,
            ids,
            url,
            phrase_list,
        )

    # ------------------------------------------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------------------------------------------

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
            "chunk_mask": batch[6],
        }

        return inputs, ex_indices, ex_phrase_numbers, phrase_list, url

    # ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def test(model, inputs, numbers, max_phrase_words):
        model.network.eval()
        with torch.no_grad():
            logits = model.network(**inputs)  # shape = (batch_size, max_diff_gram_num)

        assert (logits.shape[0] == len(numbers)) and (logits.shape[1] == max(numbers))
        logits = logits.data.cpu().tolist()

        logit_lists = [logits[batch_id][:num] for batch_id, num in enumerate(numbers)]
        return logit_lists

    # ------------------------------------------------------------------------------------------------------------
    @classmethod
    def phrase(cls, cfg, phrase_list, url, logit_lists, return_num=None):
        batch_predictions = []
        for batch_id, logit_list in enumerate(logit_lists):
            cur_url, cur_phrase_list = url[batch_id], phrase_list[batch_id]

            n_best_phrases_scores = cls.decode_n_best_candidates(
                gram_list=cur_phrase_list, score_logits=logit_list
            )
            candidate_KP, score_KP = remove_empty_phase(n_best_phrases_scores)

            if return_num:
                if cfg.evaluate.eval_stem:
                    candidate_KP, score_KP = del_stemming_duplicate_phrase(
                        candidate_KP, score_KP, return_num
                    )
                else:
                    candidate_KP = candidate_KP[:return_num]
                    score_KP = score_KP[:return_num]

            assert len(candidate_KP) == len(score_KP)
            batch_predictions.append((cur_url, candidate_KP, score_KP))

        return batch_predictions

    # ------------------------------------------------------------------------------------------------------------

    @staticmethod
    def decode_n_best_candidates(gram_list, score_logits):
        assert len(gram_list) == len(score_logits)
        ngrams = [(gram.split(), score) for gram, score in zip(gram_list, score_logits)]
        sorted_ngrams = sorted(ngrams, key=lambda x: x[1], reverse=True)

        return sorted_ngrams

    @classmethod
    def get_ngram_info_label(
        cls, doc_words, max_phrase_words, stem_flag, keyphrases=None, start_end_pos=None
    ):
        overlen_flag = False
        ngram_label = None
        # ----------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------
        tot_phrase_list, tot_mention_list = cls.get_ngram_features(
            doc_words=doc_words, max_gram=max_phrase_words, stem_flag=stem_flag
        )

        # ----------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------
        if start_end_pos is not None:
            filter_positions = utils.loader.limit_scope_length(
                start_end_pos, len(doc_words), max_phrase_words
            )

            if len(filter_positions) > 0:
                ngram_label = cls.convert_to_label(
                    filter_positions=filter_positions,
                    tot_mention_list=tot_mention_list,
                    differ_phrase_num=len(tot_phrase_list),
                )

        return tot_phrase_list, tot_mention_list, overlen_flag, ngram_label

    # ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_ngram_features(doc_words, max_gram, stem_flag=False):
        # phrase2index = TensorDict({}, batch_size=1) # use to shuffle same phrases
        phrase2index = {}  # use to shuffle same phrases
        tot_phrase_list = []  # use to final evaluation
        tot_mention_list = []  # use to train pooling the same
        gram_num = 0

        for n in range(1, max_gram + 1):
            valid_length = len(doc_words) - n + 1
            if valid_length < 1:
                break

            _mention_list = list()
            for i in range(valid_length):
                gram_num += 1
                n_gram = " ".join(doc_words[i : i + n]).lower()

                len_tot_phrase_list = (
                    len(tot_phrase_list) if tot_phrase_list is not None else 0
                )

                if stem_flag:
                    index, gram = utils.loader.whether_stem_existing(
                        n_gram, phrase2index, len_tot_phrase_list
                    )
                else:
                    index, gram = utils.loader.whether_existing(
                        n_gram, phrase2index, len_tot_phrase_list
                    )

                if gram is not None:
                    tot_phrase_list.append(gram)

                _mention_list += [index]

            tot_mention_list += [_mention_list]

        tot_phrase_list = np.array(tot_phrase_list).astype(str)
        assert len(tot_phrase_list) > 0

        assert (len(tot_phrase_list) - 1) == max(
            tot_mention_list[len(tot_mention_list) - 1]
        )
        assert (
            sum([len(_mention_list) for _mention_list in tot_mention_list]) == gram_num
        )

        return tot_phrase_list, tot_mention_list

    # ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def convert_to_label(filter_positions, tot_mention_list, differ_phrase_num):
        """First check keyphrase mentions index is same ;
        Then set keyprhase ngrams = +1  and other phrase candidates = -1 .
        """
        ngram_label = np.array([-1 for _ in range(differ_phrase_num)])

        for i in range(len(filter_positions)):
            for s, e in filter_positions[i]:
                key_index = tot_mention_list[e - s][s]
                ngram_label[key_index] = 1

        # keep have more than one positive and one negative
        if (1 in ngram_label) and (-1 in ngram_label):
            return ngram_label
        else:
            return None
