import json
import re
import string
import logging
import numpy as np
import unicodedata

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

logger = logging.getLogger()


def normalize_answer(s):
    def lower(text):
        return text.lower()

    return " ".join([lower(x) for x in s]).rstrip()


def remove_empty(a_list):
    new_list = []
    for i in a_list:
        if len(i) > 0:
            if len(i[0]) > 0:
                new_list.append(normalize_answer(i))
    return new_list


# ----------------------------------------------------------------------------
# stem phrase
def norm_and_stem(phrase_list, merge=False):
    norm_stem_phrases = list() if merge else dict()
    for phrase in phrase_list:
        norm_chars = unicodedata.normalize("NFD", phrase)
        stem_chars = " ".join([stemmer.stem(w) for w in norm_chars.split(" ")])
        if merge:
            norm_stem_phrases.append(norm_chars)
            norm_stem_phrases.append(stem_chars)
        else:
            if stem_chars not in norm_stem_phrases:
                norm_stem_phrases[stem_chars] = [norm_chars]
            else:
                norm_stem_phrases[stem_chars] += [norm_chars]

    return norm_stem_phrases


def get_match_scores(cfg, pred_list, truth_list):
    if cfg.evaluate.eval_stem:
        norm_stem_preds = norm_and_stem(pred_list)
        norm_stem_truths = norm_and_stem(truth_list)  # , merge=True)
        match_score = np.asarray([0.0] * len(norm_stem_preds), dtype="float32")
        match_truths = np.asarray([0] * len(norm_stem_truths), dtype="float32")

        for pred_id, (pred_stem, pred_norms) in enumerate(norm_stem_preds.items()):
            pred_norms_set = set(pred_norms)

            for truth_id, (truth_stem, truth_norms) in enumerate(
                norm_stem_truths.items()
            ):
                if match_truths[truth_id] != 1 and (
                    pred_stem == truth_stem
                    or pred_norms_set.intersection(set(truth_norms))
                ):
                    match_score[pred_id] = 1
                    match_truths[truth_id] = 1
        norm_stem = norm_stem_truths, norm_stem_preds
    else:
        match_score = np.asarray(
            [1 if pred_seq in truth_list else 0 for pred_seq in pred_list],
            dtype="float32",
        )

    return match_score, norm_stem


# ----------------------------------------------------------------------------


def evaluate(cfg, candidates, references, urls, max_k):
    precision_scores = dict()
    recall_scores = dict()
    f1_scores = dict()
    category_f1_scores = dict()

    category_intervals = [
        (0, 512),
        (512, 1024),
        (1024, 2048),
        (2048, 4096),
        (4096, 8192),
        8192,
    ]

    K = ["O"]
    K.extend(list(range(1, max_k + 1)))

    for k in K:
        precision_scores[k] = list()
        recall_scores[k] = list()
        f1_scores[k] = list()
        category_f1_scores[k] = {interval: list() for interval in category_intervals}

    for url in urls:
        candidate = remove_empty(
            candidates[url]["keyphrases"]
        )  # convert word list to string
        if "keyphrases" in str(references[url]):
            reference = remove_empty(references[url]["keyphrases"])  # have remove empty
        else:
            reference = remove_empty(references[url]["KeyPhrases"])  # have remove empty
        # stem match scores
        match_list, norm_stem = get_match_scores(cfg, candidate, reference)

        if cfg.evaluate.eval_stem:
            norm_stem_truths, norm_stem_preds = norm_stem
            reference = norm_stem_truths.keys()

        # Padding
        if len(match_list) < max_k:
            for _ in range(max_k - len(match_list)):
                candidate.append("")
            assert len(candidate) == max_k

        for cur_topk in f1_scores.keys():
            topk = len(reference) if cur_topk == "O" else cur_topk
            cur_match_list = match_list[:topk]

            # Micro-Averaged  Method
            micropk = (
                float(sum(cur_match_list)) / float(len(candidate[:topk]))
                if len(candidate[:topk]) > 0
                else 0.0
            )
            micrork = (
                float(sum(cur_match_list)) / float(len(reference))
                if len(reference) > 0
                else 0.0
            )
            if micrork > 1:
                micrork = 1.0

            if micropk + micrork > 0:
                microf1 = float(2 * (micropk * micrork)) / (micropk + micrork)
            else:
                microf1 = 0.0

            precision_scores[cur_topk] += [micropk]
            recall_scores[cur_topk] += [micrork]
            f1_scores[cur_topk] += [microf1]

            # Category F1
            for interval in category_intervals[:-1]:
                if references[url]["doc_words"] in range(*interval):
                    category_f1_scores[cur_topk][interval] += [microf1]
                    break
            else:
                category_f1_scores[cur_topk][category_intervals[-1]] += [microf1]

    for k in K:
        precision_scores[k] = np.mean(precision_scores[k])
        recall_scores[k] = np.mean(recall_scores[k])
        f1_scores[k] = np.mean(f1_scores[k])

        for interval in category_intervals:
            category_f1_scores[k][interval] = np.mean(category_f1_scores[k][interval])

    return f1_scores, precision_scores, recall_scores, category_f1_scores


def files_are_good(candidate, reference):
    referenceURLs = set(reference.keys())
    candidateURLs = set(candidate.keys())
    if len((referenceURLs - candidateURLs)) > 0:
        logger.info(
            "ERROR:Candidate File is missing URLS present in reference file\nMissing urls:{}".format(
                referenceURLs - candidateURLs
            )
        )
        return False
    if len((candidateURLs - referenceURLs)) > 0:
        logger.info(
            "ERROR:Candidate File includes URLS not present in reference file\nUnexpected urls:{}".format(
                candidateURLs - referenceURLs
            )
        )
        return False
    return True


def load_file(filename):
    data = {}
    with open(filename, "r") as f:
        for json_line in f:
            item = json.loads(json_line)
            data[item["url"]] = item
    return data


def evaluate_kp20k(cfg, candidate, reference_filename):
    reference = load_file(reference_filename)
    for key in reference:
        reference[key].pop("start_end_pos")
        reference[key]["doc_words"] = len(reference[key]["doc_words"])

    if files_are_good(candidate, reference) is True:
        candidate_urls = set(candidate.keys())
        reference_urls = set(reference.keys())

        urls = reference_urls.intersection(candidate_urls)
        return evaluate(cfg, candidate, reference, urls, max_k=100)
    else:
        logger.info(
            "Candidate file and Reference are not comparable. Please verify your candidate file."
        )
