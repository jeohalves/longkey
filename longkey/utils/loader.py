import os
import json
import codecs
import logging
import unicodedata
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import torch

stemmer = PorterStemmer()
logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# pre-trained model tokenize
def tokenize(doc_words, tokenizer):
    tokens = valid_mask = tok_to_orig_index = orig_to_tok_index = valid_token_ids = None

    i = 0
    while i < len(doc_words):
        sub_tokens = np.array(tokenizer.tokenize(doc_words[i]))

        if len(sub_tokens) < 1:
            sub_tokens = np.array([tokenizer.unk_token_id])

        orig_to_tok_index = (
            np.array([0])
            if orig_to_tok_index is None
            else np.append(orig_to_tok_index, len(tokens))
        )
        valid_token_ids = (
            np.array([i] + [-1] * (len(sub_tokens) - 1))
            if valid_token_ids is None
            else np.append(valid_token_ids, [i] + [-1] * (len(sub_tokens) - 1))
        )
        tok_to_orig_index = (
            np.array([i] * len(sub_tokens))
            if tok_to_orig_index is None
            else np.append(tok_to_orig_index, [i] * len(sub_tokens))
        )
        tokens = sub_tokens if tokens is None else np.append(tokens, sub_tokens)
        valid_mask = (
            np.array(range(len(sub_tokens)))
            if valid_mask is None
            else np.append(valid_mask, range(len(sub_tokens)))
        )

        i += 1

    valid_mask[valid_mask != 0] = -1
    valid_mask[valid_mask == 0] = 1
    valid_mask[valid_mask == -1] = 0

    return (
        tokens,
        torch.LongTensor(valid_mask),
        tok_to_orig_index,
        orig_to_tok_index,
        torch.LongTensor(tok_to_orig_index),
    )


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# fucntions for converting labels
def flat_rank_pos(start_end_pos):
    flatten_postions = [pos for poses in start_end_pos for pos in poses]
    sorted_positions = sorted(flatten_postions, key=lambda x: x[0])
    return sorted_positions


def strict_filter_overlap(positions):
    """delete overlap keyphrase positions."""
    previous_e = -1
    filter_positions = []
    for i, (s, e) in enumerate(positions):
        if s <= previous_e:
            continue
        filter_positions.append(positions[i])
        previous_e = e
    return filter_positions


def loose_filter_overlap(positions):
    """delete overlap keyphrase positions."""
    previous_s = -1
    filter_positions = []
    for i, (s, e) in enumerate(positions):
        if previous_s == s:
            continue
        elif previous_s < s:
            filter_positions.append(positions[i])
            previous_s = s
        else:
            logger.info("Error! previous start large than new start")
    return filter_positions


def limit_phrase_length(positions, max_phrase_words):
    filter_positions = [
        pos for pos in positions if (pos[1] - pos[0] + 1) <= max_phrase_words
    ]
    return filter_positions


def limit_scope_length(start_end_pos, valid_length, max_phrase_words):
    """filter out positions over scope & phase_length > max_phrase_words"""
    filter_positions = pd.Series()
    for positions in start_end_pos:
        _filter_position = np.array(
            [
                pos
                for pos in positions
                if pos[1] < valid_length and (pos[1] - pos[0] + 1) <= max_phrase_words
            ]
        )
        if len(_filter_position) > 0:
            filter_positions = (
                pd.Series([_filter_position])
                if len(filter_positions) == 0
                else filter_positions._append(
                    pd.Series([_filter_position]), ignore_index=True
                )
            )
    return filter_positions


def stemming(phrase):
    norm_chars = unicodedata.normalize("NFD", phrase)

    try:
        stem_chars = " ".join([stemmer.stem(w) for w in norm_chars.split(" ")])
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(f"Input: {norm_chars}")
        for w in norm_chars.split(" "):
            try:
                logger.error(f"Stem: {stemmer.stem(w)}")
            except Exception as e:
                logger.error(f"Error: {e}")
        stem_chars = norm_chars
    return norm_chars, stem_chars


def whether_stem_existing(gram, phrase2index, phrase_index):
    """If :
    unicoding(gram) and stemming(gram) not in phrase2index,
    Return : not_exist_flag
    Else :
    Return : index already in phrase2index.
    """
    norm_gram, stem_gram = stemming(gram)
    if norm_gram in phrase2index:
        index = phrase2index[norm_gram]
        phrase2index[stem_gram] = index
        return index, None

    elif stem_gram in phrase2index:
        index = phrase2index[stem_gram]
        phrase2index[norm_gram] = index
        return index, None

    else:
        index = phrase_index
        phrase2index[norm_gram] = index
        phrase2index[stem_gram] = index
        return index, gram


def whether_existing(gram, phrase2index, phrase_index):
    """If :
    gram not in phrase2index,
    Return : not_exist_flag
    Else :
    Return : index already in phrase2index.
    """
    if gram in phrase2index:
        index = phrase2index[gram]
        return index, None
    else:
        index = phrase_index
        phrase2index[gram] = index
        return index, gram
