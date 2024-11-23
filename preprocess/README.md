# Preprocessing

## Convert LDKP datasets

Load the LDKP datasets and convert them to JSON format. Here is some examples to run the conversion script:

```
python preprocess/jsonify_ldkp.py --dataset 3 --split train --output PATH/TO/DATASETS_DIR/ldkp3k/ --filename "train.json"

python preprocess/jsonify_ldkp.py --dataset 3 --split validation --output PATH/TO/DATASETS_DIR/ldkp3k/ --filename "dev.json"

python preprocess/jsonify_ldkp.py --dataset 10 --split test --output PATH/TO/DATASETS_DIR/ldkp10k/ --filename "test.json"
```

## Convert custom dataset

Convert input dataset to custom JSON format. Root directory must contain two directories: 'docsutf8' and 'keys'. Each sample must contain a .txt and a .key with same name for the respective sample. Example:

```
root_dir/
__docsutf8/
____doc0.txt
____doc1.txt
____doc2.txt
____doc3.txt
__keys/
____doc0.key
____doc1.key
____doc2.key
____doc3.key
```

The input text don't have a specific format. For the keyphrase input file, each line contains only a single keyphrase. Example:

```
breath of the wild
hyrule
a link to the past
```

See an example below to run the conversion script:

```
python preprocess/jsonify_multidata.py -i PATH/TO/INPUT_DIR/NLM500 -o PATH/TO/DATASETS_DIR/NLM500 --filename "test.json"
```
