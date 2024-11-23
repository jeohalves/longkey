import json
import os
from prepro_utils import remove_fullstop, find_sequence, clean_phrase
from tqdm import tqdm
import statistics
import argparse
from os.path import join
from datasets import load_dataset
import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def order_sections(sample):
    """
    Corrects the order in which different sections appear in the document. Resulting order is: title, abstract, other sections in the body
    """
    sections = []
    sec_text = []
    sec_bio_tags = []

    if "title" in sample["sections"]:
        title_idx = sample["sections"].index("title")
        sections.append(sample["sections"].pop(title_idx))
        sec_text.append(sample["sec_text"].pop(title_idx))
        sec_bio_tags.append(sample["sec_bio_tags"].pop(title_idx))

    if "abstract" in sample["sections"]:
        abstract_idx = sample["sections"].index("abstract")
        sections.append(sample["sections"].pop(abstract_idx))
        sec_text.append(sample["sec_text"].pop(abstract_idx))
        sec_bio_tags.append(sample["sec_bio_tags"].pop(abstract_idx))

    sections += sample["sections"]
    sec_text += sample["sec_text"]
    sec_bio_tags += sample["sec_bio_tags"]

    return sections, sec_text, sec_bio_tags


def get_tokens(tokenizer, text):
    text_tokens = list()

    if tokenizer:
        for i, token in enumerate(text.split()):
            sub_tokens = tokenizer.tokenize(token)
            if len(sub_tokens) < 1:
                sub_tokens = ["[UNK]"]

            text_tokens.extend(sub_tokens)
    else:
        text_tokens = text.split()

    return text_tokens


def main(dataset_key, split, dest_file, subset):
    overlen_kp = 0
    sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    dataset = load_dataset(f"midas/ldkp{dataset_key}k", subset)[split]

    num_words = list()

    dest_dir = os.path.abspath(dest_file)
    dest_dir = dest_dir.split("/")[:-1]
    dest_dir = "/".join(dest_dir)

    os.makedirs(dest_dir, exist_ok=True)

    with open(dest_file, "w") as jsonfile:
        sample_id = 0

        for sample in tqdm(dataset):
            sections, sec_text, sec_bio_tags = order_sections(sample)
            sec_text = [" ".join(x) for x in sec_text]

            entire_text = " ".join(sec_text)
            if len(entire_text) < 100:
                print(len(entire_text), entire_text, sections, sec_text, sec_bio_tags)
                continue
            splitted_text = sentence_tokenizer.tokenize(entire_text)
            splitted_text = [remove_fullstop(x) for x in splitted_text]
            splitted_text = [x.lower() for x in splitted_text]

            sample_id += 1

            # adding a list of keyphrases to the dictionary
            sample_keyphrase_list = []

            for cur_keyphrase in sample["extractive_keyphrases"]:
                cur_keyphrase = remove_fullstop(cur_keyphrase)
                cur_keyphrase = cur_keyphrase.lower()
                cur_keyphrase = cur_keyphrase.split(" ")
                if len(cur_keyphrase) > 5:
                    overlen_kp += 1
                sample_keyphrase_list.append(cur_keyphrase)

            cur_filename = f"sample{sample_id:07}"
            a_dict = dict()

            keyphrase_list = list()
            keyphrase_list.extend(sample_keyphrase_list)

            a_dict["doc_words"] = [x.split(" ") for x in splitted_text]
            a_dict["doc_words"] = [x for y in a_dict["doc_words"] for x in y]

            num_words += [len(a_dict["doc_words"])]
            pos_list = []

            # Find occurances of keyphrases in the passage
            for keyphrase in keyphrase_list:
                kp_pos_list = []
                ind_list = find_sequence(keyphrase, a_dict["doc_words"])

                # Some keyphrases are not detected in the passage
                if ind_list == -1:
                    keyphrase_list.remove(keyphrase)
                    continue
                else:
                    for ind in ind_list:
                        start_pos = ind
                        end_pos = ind + len(keyphrase) - 1
                        kp_pos_list.append([start_pos, end_pos])
                    pos_list.append(kp_pos_list)

            # adding the keyphrases and their positions to the dictionary
            if keyphrase_list != []:
                a_dict["keyphrases"] = clean_phrase(keyphrase_list)
                a_dict["start_end_pos"] = pos_list
                # url acts as identifier while evaluating
                a_dict["url"] = cur_filename
                # writing the dictionary to the json file
                json.dump(a_dict, jsonfile)
                jsonfile.write("\n")

        print(f"# samples: {len(num_words)}")
        print(f"max: {max(num_words)}")
        print(f"min: {min(num_words)}")
        print(f"mean: {statistics.mean(num_words)}")
        print(f"std: {statistics.stdev(num_words)}")
        print(f"OVERLEN KP: {overlen_kp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LDKP datasets to JointKPE format"
    )

    parser.add_argument("--dataset", type=str, help="LDKP dataset", choices=["3", "10"])
    parser.add_argument(
        "--split",
        type=str,
        help="Dataset split",
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--output", type=str, help="Output dir")
    parser.add_argument("--filename", "-f", type=str, default=None)

    cfg = parser.parse_cfg()
    subset = "large" if cfg.split == "train" else "small"

    filename = "multidata.json" if cfg.filename is None else cfg.filename

    dest_file = join(cfg.output, filename)

    main(cfg.dataset, cfg.split, dest_file, subset)
