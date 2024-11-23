import logging
import numpy as np
import random
import torch
from longkey import utils
import math
from longkey.utils.generator import remove_empty_phase, del_stemming_duplicate_phrase
from .AbstractData import AbstractData
import torch.nn.functional as F

logger = logging.getLogger()


class DataChunk(AbstractData):
    """build datasets for train & eval"""

    # ------------------------------------------------------------------------------------------------------------
    def get_example(self, index, check_others=True, eval=False):
        cur_example = self.get_arrow_row_by_id(self.table, index)

        url, tokens, valid_mask, s_label, e_label, doc_words = self._preprocessor(
            ex=cur_example,
            tokenizer=self.tokenizer,
            max_token=self.max_token,
            mode=self.mode,
            max_phrase_words=self.cfg.model.max_phrase_words,
        )

        while check_others and url is None:
            url, tokens, valid_mask, s_label, e_label, doc_words = self.get_example(
                random.randint(0, self.__len__() - 1), check_others=False
            )

        if eval:
            return url, doc_words
        else:
            return url, tokens, valid_mask, s_label, e_label, doc_words

    # ------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index):
        url, tokens, valid_mask, s_label, e_label, doc_words = self.get_example(index)

        src_tensor, valid_mask, orig_doc_len = self._converter(
            tokens,
            valid_mask,
            self.tokenizer,
        )

        if self.mode == "train":
            return (
                index,
                src_tensor,
                valid_mask,
                s_label,
                e_label,
                orig_doc_len,
                self.max_phrase_words,
                url,
            )
        else:
            return (
                index,
                src_tensor,
                valid_mask,
                orig_doc_len,
                self.max_phrase_words,
                url,
                doc_words,
            )

    # ------------------------------------------------------------------------------------------------------------
    @classmethod
    def _preprocessor(cls, ex, tokenizer, max_token, mode, max_phrase_words):
        url = tokens = valid_mask = s_label = e_label = start_end_pos = None

        tokens, valid_mask, tok_to_orig_index, _, _ = utils.loader.tokenize(
            doc_words=ex["doc_words"], tokenizer=tokenizer
        )

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
        assert sum(valid_mask) == len(doc_words)

        if mode == "train":
            start_end_pos = ex["start_end_pos"]
            # ------------------------------------------------
            s_label, e_label, _ = cls.get_ngram_label(
                len(doc_words), start_end_pos, max_phrase_words
            )

            if len(s_label) == 0:
                url = None

        # ---------------------------------------------------------------------------

        return url, tokens, valid_mask, s_label, e_label, doc_words

    # ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _converter(tokens, valid_mask, tokenizer):
        """convert each batch data to tensor ; add [CLS] [SEP] tokens ;"""
        BOS_TOKEN = tokenizer.bos_token if tokenizer.bos_token else "[CLS]"
        src_tokens = np.append([BOS_TOKEN], tokens)
        valid_ids = np.append([0], valid_mask)

        src_tensor = torch.LongTensor(tokenizer.convert_tokens_to_ids(src_tokens))
        valid_mask = torch.LongTensor(valid_ids)
        orig_doc_len = sum(valid_ids)

        return src_tensor, valid_mask, orig_doc_len

    # ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def batchify_train(batch, encoder_output_dim, global_attention):
        """train dataloader & eval dataloader ."""

        ids = [ex[0] for ex in batch]
        docs = [ex[1] for ex in batch]
        valid_mask = [ex[2] for ex in batch]
        s_label_list = [ex[3] for ex in batch]
        e_label_list = [ex[4] for ex in batch]
        doc_word_lens = [ex[5] for ex in batch]
        max_phrase_words = [ex[6] for ex in batch][0]

        max_word_len = max([word_len for word_len in doc_word_lens])  # word-level

        # ---------------------------------------------------------------
        # [1] [2] src tokens tensor
        doc_max_length = max([d.size(0) for d in docs])
        input_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
        input_mask = torch.LongTensor(len(docs), doc_max_length).zero_()
        global_attention_mask = torch.LongTensor(len(docs), doc_max_length).zero_() if global_attention else None

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
        # [4] active mask : for n-gram
        max_ngram_length = sum([max_word_len - n for n in range(max_phrase_words)])
        active_mask = torch.LongTensor(len(docs), max_ngram_length).zero_()

        for batch_i, word_len in enumerate(doc_word_lens):
            pad_len = max_word_len - word_len

            batch_mask = []
            for n in range(max_phrase_words):
                ngram_len = word_len - n
                if ngram_len > 0:
                    gram_list = [1 for _ in range(ngram_len)] + [
                        0 for _ in range(pad_len)
                    ]
                else:
                    gram_list = [0 for _ in range(max_word_len - n)]
                batch_mask.extend(gram_list)
            active_mask[batch_i].copy_(torch.LongTensor(batch_mask))

        # -------------------------------------------------------------------
        # [5] label : for n-gram
        # 1. empty label list
        label_list = []
        for _ in range(len(docs)):
            batch_label = []
            for n in range(max_phrase_words):
                batch_label.append(
                    torch.LongTensor([0 for _ in range(max_word_len - n)])
                )
            label_list.append(batch_label)

        # 2. valid label list
        for batch_i in range(len(docs)):
            for s, e in zip(s_label_list[batch_i], e_label_list[batch_i]):
                gram = e - s
                label_list[batch_i][gram][s] = 1

        # 3. label tensor
        ngram_label = torch.LongTensor(len(docs), max_ngram_length).zero_()
        for batch_i, label in enumerate(label_list):
            ngram_label[batch_i].copy_(torch.cat(label))

        # 4. valid output
        valid_output = torch.zeros(len(docs), max_word_len, encoder_output_dim)
        return (
            input_ids,
            input_mask,
            global_attention_mask,
            valid_ids,
            active_mask,
            valid_output,
            ngram_label,
            ids,
        )

    # ------------------------------------------------------------------------------------------------------------

    @staticmethod
    def batchify_test(batch, encoder_output_dim, global_attention):
        """test dataloader for Dev & Public_Valid."""

        ids = [ex[0] for ex in batch]
        docs = [ex[1] for ex in batch]
        valid_mask = [ex[2] for ex in batch]
        doc_word_lens = [ex[3] for ex in batch]
        max_phrase_words = [ex[4] for ex in batch][0]
        url = [ex[5] for ex in batch]
        phrase_list = [ex[6] for ex in batch]

        max_word_len = max([word_len for word_len in doc_word_lens])  # word-level

        # ---------------------------------------------------------------
        # [1] [2] src tokens tensor
        doc_max_length = max([d.size(0) for d in docs])
        input_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
        input_mask = torch.LongTensor(len(docs), doc_max_length).zero_()
        global_attention_mask = torch.LongTensor(len(docs), doc_max_length).zero_() if global_attention else None

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
        # [4] active mask : for n-gram
        max_ngram_length = sum([max_word_len - n for n in range(max_phrase_words)])
        active_mask = torch.LongTensor(len(docs), max_ngram_length).zero_()

        for batch_i, word_len in enumerate(doc_word_lens):
            pad_len = max_word_len - word_len

            batch_mask = []
            for n in range(max_phrase_words):
                ngram_len = word_len - n
                if ngram_len > 0:
                    gram_list = [1 for _ in range(ngram_len)] + [
                        0 for _ in range(pad_len)
                    ]
                else:
                    gram_list = [0 for _ in range(max_word_len - n)]
                batch_mask.extend(gram_list)
            active_mask[batch_i].copy_(torch.LongTensor(batch_mask))

        # ---------------------------------------------------------------
        # [5] valid output
        valid_output = torch.zeros(len(docs), max_word_len, encoder_output_dim)
        return (
            input_ids,
            input_mask,
            global_attention_mask,
            valid_ids,
            active_mask,
            valid_output,
            doc_word_lens,
            ids,
            url,
            phrase_list,
        )

    # ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def test(model, inputs, lengths, max_phrase_words):
        model.network.eval()
        with torch.no_grad():
            logits = model.network(**inputs)
            logits = F.softmax(logits, dim=-1)
        logits = logits.data.cpu()  # [:,1]
        logits = logits.tolist()

        logit_lists = []
        sum_len = 0
        for length in lengths:
            batch_logit = []
            for n in range(max_phrase_words):
                batch_logit.append(logits[sum_len : sum_len + length - n])
                sum_len = sum_len + (length - n)
            logit_lists.append(batch_logit)
        return logit_lists

    # ------------------------------------------------------------------------------------------------------------
    @classmethod
    def phrase(
        cls,
        cfg,
        url,
        phrase_list,
        logit_lists,
        return_num,
    ):
        batch_predictions = []
        for batch_id, logit_list in enumerate(logit_lists):
            cur_url, cur_phrase_list = url[batch_id], phrase_list[batch_id]
            n_best_phrases_scores = cls.decode_n_best_candidates(
                orig_tokens=cur_phrase_list,
                gram_logits=logit_list,
                max_gram=cfg.model.max_phrase_words,
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
    def get_ngram_label(valid_length, start_end_pos, max_phrase_words):
        # flatten, rank, filter overlap for answer positions

        sorted_positions = utils.loader.flat_rank_pos(start_end_pos)
        filter_positions = utils.loader.limit_phrase_length(
            sorted_positions, max_phrase_words
        )

        if len(filter_positions) != len(sorted_positions):
            overlen_flag = True
        else:
            overlen_flag = False

        s_label, e_label = [], []
        for s, e in filter_positions:
            if e < valid_length:
                s_label.append(s)
                e_label.append(e)
            else:
                break
        assert len(s_label) == len(e_label)

        s_label = np.array(s_label)
        e_label = np.array(e_label)

        return s_label, e_label, overlen_flag

    # ------------------------------------------------------------------------------------------------------------
    @classmethod
    def decode_n_best_candidates(cls, orig_tokens, gram_logits, max_gram):
        """
        max_gram :  type :int , max_phrase_words
        return : phrase token list & score list
        """
        orig_tokens = [token.lower() for token in orig_tokens]
        sorted_ngrams = cls.decode_ngram(
            orig_tokens=orig_tokens, gram_logits=gram_logits, max_gram=max_gram
        )
        return sorted_ngrams

    # ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def decode_ngram(orig_tokens, gram_logits, max_gram):
        ngram_score = []
        for n in range(max_gram):
            for i in range(len(orig_tokens) - n):
                ngram_score.append(
                    (" ".join(orig_tokens[i : i + n + 1]), gram_logits[n][i])
                )

        phrase_set = {}
        for n_gram, n_gram_score in ngram_score:
            if n_gram not in phrase_set or n_gram_score > phrase_set[n_gram]:
                phrase_set[n_gram] = n_gram_score
            else:
                continue

        phrase_list = []
        for phrase, score in phrase_set.items():
            phrase_list.append((phrase.split(), score))

        sorted_phrase_list = sorted(phrase_list, key=lambda x: x[1], reverse=True)
        return sorted_phrase_list
