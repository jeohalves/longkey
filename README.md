# LongKey

This repository contains the code for the paper "LongKey: Keyphrase Extraction for Long Documents". The paper is available [here](). Below are the instructions to use the code.

## Roadmap

The following is a list of features that we plan to implement in the future:

- [ ] Improve the documentation
- [ ] Provide a simple way to perform inference on a single document
- [ ] Implement support for different optimizers and schedulers
- [ ] Support for pip package
- [ ] Enable HuggingFace integration for weight loading
- [ ] Improve CPU support
- [ ] Add support for BIO format-based datasets
- [ ] Incorporate other state-of-the-art (SOTA) methods
- [ ] Add multilingual support

## Installation

```
conda create --name longkey python=3.12.7
conda activate longkey
```

```
pip install -r requirements.txt
```

## Datasets

If you want to use the preprocessed datasets, please download them from [here](https://www.dropbox.com/scl/fo/y1rogglqyxfc3fj2osxv1/ACNpcsGhYhvE2iFotccdGQM?rlkey=9flmvgyjb4c4ag583lqulga57&st=yjbexo12&dl=0).
You can also follow the instructions to preprocess the original datasets [here](preprocess/README.md).

## Training

Training can be executed using the train.py script. For example, we can train the LongKey approach as follows:

```
python train.py +exp=ldkp3k dir.data=PATH/TO/DATASETS_DIR data.dataset=ldkp3k dir.exp=PATH_TO/EXPERIMENTS_DIR model.method=longkey
```

The script will create a new subdirectory in the *dir.exp* directory with the a generated name for the experiment. The checkpoints and logs will be stored in this directory. If you want to specify the name of the experiment, you can use the *exp.name* argument. Example:

```
python train.py +exp=ldkp3k dir.data=PATH/TO/DATASETS_DIR data.dataset=ldkp3k dir.exp=PATH_TO/EXPERIMENTS_DIR model.method=longkey exp.name=EXP_NAME
```

Distributed training is also supported. You can use multiple GPUs by appending the following arguments to the command:

```
distributed=true runtime.ddp=true
```

It is also important to mention that some arguments are affected by how many GPUs are being used. For example, *model.train.max_steps_per_epoch* is the number of steps per epoch regardless of the number of GPUs, so it should be divided by the number of GPUs if you are using more than one. The same applies to the *model.train.batch_size* or *model.gradient_accumulation_steps*. By default, the *batch size* is 1 and the *gradient accumulation steps* is 16. If you are using 4 GPUs, you should set the *gradient accumulation steps* to 4 (or, if you have enough memory, set the *batch size* to 4 and the *gradient accumulation steps* to 1).

You can check other possible arguments in the config directory. The *+exp* argument is used to overwrite the default values of the config file (*base.yaml*). The current options are:

```
+exp=ldkp3k
+exp=ldkp10k
+exp=bert_ldkp3k
+exp=bert_ldkp10k
```

If you want to use a custom config file, put it the absolute path to it instead of one of the above options. You can also use the *model.method* argument to specify the method to be used. The current options are:

```
model.method=longkey
model.method=joint
model.method=rank
model.method=chunk
model.method=span
model.method=tag
model.method=hypermatch
```

## Testing

Testing script is similar, except that a *config.yaml* must be inside the experiment folder (e.g., *PATH_TO/EXPERIMENTS_DIR/EXP_NAME/config.yaml*). So, the *+exp* argument is not used. Example:

```
python test.py dir.data=PATH/TO/DATASETS_DIR data.dataset=ldkp3k dir.exp=PATH_TO/EXPERIMENTS_DIR exp.name=EXP_NAME
```

The checkpoints are loaded from the experiment folder if specified in the *config.yaml* file (always inside the checkpoint folder). If you want to specify the path to the checkpoint, you can use the *load_checkpoint* argument. Example:

```
python test.py dir.data=PATH/TO/DATASETS_DIR data.dataset=ldkp3k dir.exp=PATH_TO/EXPERIMENTS_DIR load_checkpoint=PATH_TO_CHECKPOINT
```

You can use multiple GPUs for testing by setting the *distributed* argument equal to *true*.

### Evaluate multiple datasets

You can evaluate multiple datasets at once by using the *--multirun* option and changing the *data.dataset* argument, where each dataset is separated by a comma. Example with two datasets:

```
python test.py --multirun dir.data=PATH/TO/DATASETS_DIR data.dataset=ldkp3k,ldkp10k dir.exp=PATH_TO/EXPERIMENTS_DIR exp.name=EXP_NAME
```

### Weights

If you want to use the pre-trained weights, please download them from [here](https://www.dropbox.com/scl/fo/kn08j9po6yi0uxbgs3bvf/ADKicMiqf_sNKWSZy74_uvA?rlkey=6wnseu90wf8h1gpu8w3hgjs90&st=xf04i4qz&dl=0).

## Feedback

If you have any questions or suggestions, please feel free to open an issue. We will be happy to help you!

## Acknowledgements

Our implementation is based on [BERT-KPE](https://github.com/thunlp/BERT-KPE) and [HyperMatch](https://github.com/MySong7NLPer/HyperMatch) codes. Thanks for their work!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code, please cite our paper:

```
```