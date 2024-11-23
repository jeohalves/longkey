import json
import os
from prepro_utils import remove_fullstop, find_sequence, clean_phrase
from tqdm import tqdm
import statistics
import argparse
from os.path import join


def get_bert_tokens(tokenizer, text):
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


def main(passage_folder, keyphrase_folder, dest_file):
    file_list = os.listdir(passage_folder)
    file_list.sort()

    num_words = list()

    dest_dir = os.path.abspath(dest_file)
    dest_dir = dest_dir.split("/")[:-1]
    dest_dir = "/".join(dest_dir)

    os.makedirs(dest_dir)

    with open(dest_file, "w") as jsonfile:
        # for file_name in file_list:
        for file_name in tqdm(file_list):
            with open(passage_folder + file_name, "r") as f:
                text = f.read()
                # removing fullstops
                text = remove_fullstop(text)
                text = text.lower()
                str_en = text.encode("ascii", "ignore")
                entire_text = str_en.decode()
                # adding a list of words to the dictionary
                splitted_text = entire_text.split()

            # adding a list of keyphrases to the dictionary
            base = file_name[:-4]
            sample_keyphrase_list = []
            with open(keyphrase_folder + base + ".key", "r") as f:
                text = f.readlines()

                for line in text:
                    line = line.strip("\n")
                    if line:
                        line = line.lower()
                        line = remove_fullstop(line)
                        str_en = line.encode("ascii", "ignore")
                        str_de = str_en.decode()
                        sample_keyphrase_list.append(str_de.split())

            a_dict = dict()

            keyphrase_list = list()
            keyphrase_list.extend(sample_keyphrase_list)

            a_dict["doc_words"] = list()
            a_dict["doc_words"].extend(splitted_text)

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
                a_dict["url"] = file_name
                # writing the dictionary to the json file
                json.dump(a_dict, jsonfile)
                jsonfile.write("\n")

        print(f"# samples: {len(num_words)}")
        print(f"max: {max(num_words)}")
        print(f"min: {min(num_words)}")
        print(f"mean: {statistics.mean(num_words)}")
        print(f"std: {statistics.stdev(num_words)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert docsutf8 and keys to JointKPE format"
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Root directory. Must contain the 'docsutf8' and 'keys' directories.",
    )
    parser.add_argument("--output", "-o", type=str, help="Output dir")
    parser.add_argument("--filename", "-f", type=str, default=None)

    cfg = parser.parse_cfg()

    passage_folder = join(cfg.input, "docsutf8/")
    keyphrase_folder = join(cfg.input, "keys/")

    filename = "multidata.json" if cfg.filename is None else cfg.filename
    dest_file = join(cfg.output, filename)

    main(passage_folder, keyphrase_folder, dest_file)
