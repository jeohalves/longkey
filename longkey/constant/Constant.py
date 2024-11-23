PAD = 0
UNK = 100
BOS = 101
EOS = 102
DIGIT = 1

PAD_WORD = "[PAD]"
UNK_WORD = "[UNK]"
BOS_WORD = "[CLS]"
EOS_WORD = "[SEP]"
DIGIT_WORD = "DIGIT"

Idx2Tag = ["O", "B", "I", "E", "U"]
Tag2Idx = {"O": 0, "B": 1, "I": 2, "E": 3, "U": 4}

POS_TAGS = {
    "Other": 0,
    "CC": 1,
    "CD": 2,
    "DT": 3,
    "EX": 4,
    "FW": 5,
    "IN": 6,
    "JJ": 7,
    "JJR": 8,
    "JJS": 9,
    "LS": 10,
    "MD": 11,
    "NN": 12,
    "NNS": 13,
    "NNP": 14,
    "NNPS": 15,
    "PDT": 16,
    "POS": 17,
    "PRP": 18,
    "PRP$": 19,
    "RB": 20,
    "RBR": 21,
    "RBS": 22,
    "RP": 23,
    "SYM": 24,
    "TO": 25,
    "UH": 26,
    "VB": 27,
    "VBD": 28,
    "VBG": 29,
    "VBN": 30,
    "VBP": 31,
    "VBZ": 32,
    "WDT": 33,
    "WP": 34,
    "WP$": 35,
    "WRB": 36,
}


class IdxTag_Converter(object):
    """idx2tag : a tag list like ['O','B','I','E','U']
    tag2idx : {'O': 0, 'B': 1, ..., 'U':4}
    """

    def __init__(self, idx2tag):
        self.idx2tag = idx2tag
        tag2idx = {}
        for idx, tag in enumerate(idx2tag):
            tag2idx[tag] = idx
        self.tag2idx = tag2idx

    def convert_idx2tag(self, index_list):
        tag_list = [self.idx2tag[index] for index in index_list]
        return tag_list

    def convert_tag2idx(self, tag_list):
        index_list = [self.tag2idx[tag] for tag in tag_list]
        return index_list


# 'O' : non-keyphrase
# 'B' : begin word of the keyphrase
# 'I' : middle word of the keyphrase
# 'E' : end word of the keyphrase
# 'U' : single word keyphrase
