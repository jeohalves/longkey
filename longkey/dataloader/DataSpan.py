import logging
import numpy as np
import random
import torch
from longkey import utils
import math
from longkey.utils.generator import remove_empty_phase, del_stemming_duplicate_phrase
from .AbstractData import AbstractData

logger = logging.getLogger()


class DataSpan(AbstractData):
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
            s_label = torch.LongTensor(s_label)
            e_label = torch.LongTensor(e_label)

            return (
                index,
                src_tensor,
                valid_mask,
                orig_doc_len,
                s_label,
                e_label,
                url,
            )
        else:
            return (
                index,
                src_tensor,
                valid_mask,
                orig_doc_len,
                url,
                doc_words,
            )

    @classmethod
    def _preprocessor(cls, ex, tokenizer, max_token, mode, max_phrase_words):
        url = tokens = valid_mask = s_label = e_label = start_end_pos = None

        # tokenize
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

            s_label, e_label = cls.get_span_label(start_end_pos, len(doc_words))

            if s_label is None or len(s_label) == 0:
                url = None

        return url, tokens, valid_mask, s_label, e_label, doc_words

    @staticmethod
    def _converter(tokens, valid_mask, tokenizer):
        BOS_TOKEN = tokenizer.bos_token if tokenizer.bos_token else "[CLS]"
        src_tokens = np.append([BOS_TOKEN], tokens)
        valid_ids = np.append([0], valid_mask)

        src_tensor = torch.LongTensor(tokenizer.convert_tokens_to_ids(src_tokens))
        valid_mask = torch.LongTensor(valid_ids)
        orig_doc_len = sum(valid_ids)

        return src_tensor, valid_mask, orig_doc_len

    @staticmethod
    def batchify_train(batch, encoder_output_dim, global_attention):
        """train dataloader & eval dataloader ."""

        ids = [ex[0] for ex in batch]
        docs = [ex[1] for ex in batch]
        valid_mask = [ex[2] for ex in batch]
        doc_word_lens = [ex[3] for ex in batch]
        s_label_list = [ex[4] for ex in batch]
        e_label_list = [ex[5] for ex in batch]

        max_word_len = max([word_len for word_len in doc_word_lens])  # word-level

        # ---------------------------------------------------------------
        # [1][2]src tokens tensor
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
        # [4] start label [5] active_mask
        s_label = torch.LongTensor(len(s_label_list), max_word_len).zero_()
        active_mask = torch.LongTensor(len(s_label_list), max_word_len).zero_()
        for i, s in enumerate(s_label_list):
            s_label[i, : s.size(0)].copy_(s)
            active_mask[i, : s.size(0)].fill_(1)

        # ---------------------------------------------------------------
        # [6] end label [7] end_mask
        e_label_max_length = max([e.size(0) for e in e_label_list])
        e_label = torch.LongTensor(len(e_label_list), e_label_max_length).zero_()
        end_mask = torch.LongTensor(len(e_label_list), e_label_max_length).zero_()
        for i, e in enumerate(e_label_list):
            if e.size(0) <= 0:
                continue
            e_label[i, : e.size(0)].copy_(e)
            end_mask[i, : e.size(0)].fill_(1)

        # -------------------------------------------------------------------
        # [8] Empty Tensor : word-level max_len
        valid_output = torch.zeros(len(docs), max_word_len, encoder_output_dim)
        return (
            input_ids,
            input_mask,
            global_attention_mask,
            valid_ids,
            valid_output,
            active_mask,
            s_label,
            e_label,
            end_mask,
            ids,
        )

    @staticmethod
    def batchify_test(batch, encoder_output_dim, global_attention):
        """test dataloader for Dev & Public_Valid."""

        ids = [ex[0] for ex in batch]
        docs = [ex[1] for ex in batch]
        valid_mask = [ex[2] for ex in batch]
        doc_word_lens = [ex[3] for ex in batch]
        url = [ex[4] for ex in batch]
        phrase_list = [ex[5] for ex in batch]

        max_word_len = max([word_len for word_len in doc_word_lens])  # word-level

        # ---------------------------------------------------------------
        # [1][2]src tokens tensor
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
        # [4] valid length tensor
        active_mask = torch.LongTensor(len(doc_word_lens), max_word_len).zero_()
        for i, cur_l in enumerate(doc_word_lens):
            active_mask[i, :cur_l].fill_(1)

        # -------------------------------------------------------------------
        # [5] Empty Tensor : word-level max_len
        valid_output = torch.zeros(len(docs), max_word_len, encoder_output_dim)
        return (
            input_ids,
            input_mask,
            global_attention_mask,
            valid_ids,
            valid_output,
            active_mask,
            doc_word_lens,
            ids,
            url,
            phrase_list,
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
            "valid_output": batch[4],
            "active_mask": batch[5],
            "s_label": batch[6],
            "e_label": batch[7],
            "end_mask": batch[8],
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
            "valid_output": batch[4],
            "active_mask": batch[5],
        }

        return inputs, ex_indices, ex_phrase_numbers, phrase_list, url

    @staticmethod
    def test(model, inputs, lengths, max_phrase_words):
        model.network.eval()
        with torch.no_grad():
            s_logits, e_logits = model.network(**inputs)

        assert s_logits.size(0) == e_logits.size(0) == sum(lengths)
        s_logits = s_logits.data.cpu().tolist()
        e_logits = e_logits.data.cpu()

        start_lists, end_lists = [], []
        sum_len = 0
        for length in lengths:
            start_lists.append(s_logits[sum_len : (sum_len + length)])
            end_lists.append(e_logits[sum_len : (sum_len + length), :length].tolist())
            sum_len = sum_len + length
        return (start_lists, end_lists)

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
        for batch_id, (start_logit, end_logit) in enumerate(
            zip(logit_lists[0], logit_lists[1])
        ):
            cur_url, cur_phrase_list = url[batch_id], phrase_list[batch_id]

            n_best_phrases_scores = cls.decode_n_best_candidates(
                orig_tokens=cur_phrase_list,
                start_logit=start_logit,
                end_logit=end_logit,
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

    # -------------------------------------------------------------------------------------------
    # preprocess label
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def get_span_label(start_end_pos, max_doc_word):
        # flatten, rank, filter overlap for answer positions
        sorted_positions = utils.loader.flat_rank_pos(start_end_pos)
        filter_positions = utils.loader.loose_filter_overlap(sorted_positions)

        s_label = [0] * max_doc_word
        e_label = []
        for s, e in filter_positions:
            if (s <= e) and (e < max_doc_word):
                s_label[s] = 1
                e_label.append(e)
            else:
                continue

        if (len(e_label) > 0) and (sum(s_label) == len(e_label)):
            return s_label, e_label
        else:
            return None, None

    @classmethod
    def decode_n_best_candidates(cls, orig_tokens, start_logit, end_logit, max_gram):
        """
        max_gram :  type :int , max_phrase_words
        return : phrase token list & score list
        """
        assert len(orig_tokens) == len(start_logit) == len(end_logit)
        orig_tokens = [token.lower() for token in orig_tokens]

        sorted_ngrams = cls.decode_span2phrase(
            **{
                "orig_tokens": orig_tokens,
                "start_logit": start_logit,
                "end_logit": end_logit,
                "max_gram": max_gram,
            }
        )
        return sorted_ngrams

    @staticmethod
    def decode_span2phrase(orig_tokens, start_logit, end_logit, max_gram):
        phrase2score = {}
        for i, s in enumerate(start_logit):
            for j, e in enumerate(end_logit[i][i : (i + max_gram)]):
                phrase = " ".join(orig_tokens[i : (i + j + 1)])
                score = s * e
                if (phrase not in phrase2score) or (score > phrase2score[phrase]):
                    phrase2score[phrase] = score
                else:
                    continue

        phrase_list = []
        for phrase, score in phrase2score.items():
            phrase_list.append((phrase.split(), score))

        sorted_phrase_list = sorted(phrase_list, key=lambda x: x[1], reverse=True)
        return sorted_phrase_list
