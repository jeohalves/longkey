import string
import logging
from nltk.stem.porter import PorterStemmer
import re

stemmer = PorterStemmer()
logger = logging.getLogger()


def remove_fullstop(text):
    """removing just fullstops from the text"""
    regex = re.compile("[%s]+" % re.escape(string.punctuation))
    text = regex.sub("", text)
    return re.sub(" +", " ", text).strip()


def find_sequence(seq, _list):
    """Finding all the indices of the keyphrases in the passage"""
    if not seq:
        return -1

    seq_list = seq
    all_occurrence = [
        idx
        for idx in [i for i, x in enumerate(_list) if x == seq_list[0]]
        if seq_list == _list[idx : idx + len(seq_list)]
    ]
    return -1 if not all_occurrence else all_occurrence


# ----------------------------------------------------------------------------------------
# clean and refactor source OpenKP dataset
# ----------------------------------------------------------------------------------------


def clean_phrase(Keyphrases):
    """remove empty, duplicate and punctuation, do lower case"""

    def lower(text):
        return text.lower()

    def remove_punc(words):
        strings = " ".join(words)
        return strings.translate(str.maketrans("", "", string.punctuation))

    phrase_set = set()
    return_phrases = []
    for phrase in Keyphrases:
        if len(phrase) > 0:
            clean_phrase = lower(remove_punc(phrase))
            if len(clean_phrase) > 0 and clean_phrase not in phrase_set:
                return_phrases.append(clean_phrase.split())
                phrase_set.add(clean_phrase)
    return return_phrases
