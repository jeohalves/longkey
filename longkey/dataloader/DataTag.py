import logging
import numpy as np
import random
import torch
from longkey import utils
import math
from longkey.utils.generator import remove_empty_phase, del_stemming_duplicate_phrase
from .AbstractData import AbstractData
import torch.nn.functional as F
from ..constant import Tag2Idx

logger = logging.getLogger()


class DataTag(AbstractData):
    """build datasets for train & eval"""

    def get_example(self, index, check_others=True, eval=False):
        cur_example = self.get_arrow_row_by_id(self.table, index)

        (url, tokens, valid_mask, doc_words, label) = self._preprocessor(
            ex=cur_example,
            tokenizer=self.tokenizer,
            max_token=self.max_token,
            mode=self.mode,
        )

        while check_others and url is None:
            (url, tokens, valid_mask, doc_words, label) = self.get_example(
                random.randint(0, self.__len__() - 1), check_others=False
            )

        return (url, tokens, valid_mask, doc_words, label)

    def __getitem__(self, index):
        (
            url,
            tokens,
            valid_mask,
            doc_words,
            label,
        ) = self.get_example(index)

        src_tensor, valid_mask, orig_doc_len = self._converter(
            tokens,
            valid_mask,
            self.tokenizer,
        )

        if self.mode == "train":
            label = torch.LongTensor(label)

            return (
                index,
                src_tensor,
                valid_mask,
                orig_doc_len,
                label,
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
    def _preprocessor(cls, ex, tokenizer, max_token, mode):
        tokens = valid_mask = start_end_pos = label = url = doc_words = None

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

        if mode == "train":
            doc_length = len(ex["doc_words"])
            start_end_pos = ex["start_end_pos"]
            # ------------------------------------------------
            label_dict = cls.get_tag_label(start_end_pos, doc_length)
            label = label_dict["label"][:max_word]
            assert sum(valid_mask) == len(label)

        return (
            url,
            tokens,
            valid_mask,
            doc_words,
            label,
        )

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

    @staticmethod
    def batchify_train(batch, encoder_output_dim, global_attention):
        """train dataloader & eval dataloader ."""

        ids = [ex[0] for ex in batch]
        docs = [ex[1] for ex in batch]
        valid_mask = [ex[2] for ex in batch]
        doc_word_lens = [ex[3] for ex in batch]
        label_list = [ex[4] for ex in batch]
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
        # label tensor
        labels = torch.LongTensor(len(label_list), max_word_len).zero_()
        active_mask = torch.LongTensor(len(label_list), max_word_len).zero_()
        for i, t in enumerate(label_list):
            labels[i, : t.size(0)].copy_(t)
            active_mask[i, : t.size(0)].fill_(1)

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
            labels,
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
        # valid length tensor
        active_mask = torch.LongTensor(len(doc_word_lens), max_word_len).zero_()
        for i, cur_len in enumerate(doc_word_lens):
            active_mask[i, :cur_len].fill_(1)

        # -------------------------------------------------------------------
        # [4] Empty Tensor : word-level max_len
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
    def test(model, inputs, lengths, max_phrase_words):
        model.network.eval()
        with torch.no_grad():
            logits = model.network(**inputs)
            logits = F.softmax(logits, dim=-1)
        logits = logits.data.cpu().tolist()
        assert len(logits) == sum(lengths)

        logit_lists = []
        sum_len = 0
        for length in lengths:
            logit_lists.append(logits[sum_len : sum_len + length])
            sum_len = sum_len + length
        return logit_lists

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
                token_logits=logit_list,
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

    @classmethod
    def decode_n_best_candidates(cls, orig_tokens, token_logits, max_gram):
        """
        max_gram :  type :int , max_phrase_words
        return : phrase token list & score list
        """
        assert len(orig_tokens) == len(token_logits)
        orig_tokens = [token.lower() for token in orig_tokens]

        ngrams = []
        for n in range(1, max_gram + 1):
            ngrams.extend(cls.decode_ngram(orig_tokens, token_logits, n))
        # sorted all n-grams
        sorted_ngrams = sorted(ngrams, key=lambda x: x[1], reverse=True)
        return sorted_ngrams

    @staticmethod
    def decode_ngram(orig_tokens, token_logits, n):
        """
        Combine n-gram score and sorted
        Inputs :
            n : n_gram
            orig_tokens : document lower cased words' list
            token_logits : each token has five score : for 'O', 'B', 'I', 'E', 'U' tag
            sum_tf : if True Sum All Mention
        Outputs : sorted phrase and socre list
        """
        if n == 1:
            ngram_ids = [Tag2Idx["U"]]
        elif n >= 2:
            ngram_ids = (
                [Tag2Idx["B"]] + [Tag2Idx["I"] for _ in range(n - 2)] + [Tag2Idx["E"]]
            )
        else:
            logger.info("invalid %d-gram !" % n)
        offsets = [i for i in range(len(ngram_ids))]

        # combine n-gram scores
        phrase_set = {}
        valid_length = len(orig_tokens) - n + 1
        for i in range(valid_length):
            n_gram = " ".join(orig_tokens[i : i + n])
            n_gram_score = min(
                [token_logits[i + bias][tag] for bias, tag in zip(offsets, ngram_ids)]
            )

            if n_gram not in phrase_set or n_gram_score > phrase_set[n_gram]:
                phrase_set[n_gram] = n_gram_score
            else:
                continue

        phrase_list = []
        for phrase, score in phrase_set.items():
            phrase_list.append((phrase.split(), score))

        sorted_phrase_list = sorted(phrase_list, key=lambda x: x[1], reverse=True)
        return sorted_phrase_list

    # -------------------------------------------------------------------------------------------
    # preprocess label
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def get_tag_label(start_end_pos, doc_length):
        # flatten, rank, filter overlap for answer positions
        sorted_positions = utils.loader.flat_rank_pos(start_end_pos)
        filter_positions = utils.loader.strict_filter_overlap(sorted_positions)

        if len(filter_positions) != len(sorted_positions):
            overlap_flag = True
        else:
            overlap_flag = False

        label = [Tag2Idx["O"]] * doc_length
        for s, e in filter_positions:
            if s == e:
                label[s] = Tag2Idx["U"]

            elif (e - s) == 1:
                label[s] = Tag2Idx["B"]
                label[e] = Tag2Idx["E"]

            elif (e - s) >= 2:
                label[s] = Tag2Idx["B"]
                label[e] = Tag2Idx["E"]
                for i in range(s + 1, e):
                    label[i] = Tag2Idx["I"]
            else:
                logger.info("ERROR")
                break
        return {"label": label, "overlap_flag": overlap_flag}
