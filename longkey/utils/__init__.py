from genericpath import isfile
import time
import json
from omegaconf import OmegaConf
import torch
import random
import argparse
import logging
import numpy as np
import math
import os
from os.path import join
import sys

from . import loader
from . import generator


logger = logging.getLogger()

def init_cfg_config(cfg, run_mode, override_args):

    # setting env configs
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["NCCL_IB_GID_INDEX"] = "3"
    os.environ["TORCH_CUDNN_V8_LRU_CACHE_LIMIT"] = "0"
    os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"

    # Result folder
    if cfg.exp.name is not None:
        save_folder = join(cfg.dir.exp, cfg.exp.name)
    else:
        formatted_pretrain_model = cfg.model.pretrain_model.replace("/", "_")
        folder_name = f"{run_mode}_{formatted_pretrain_model.split('-')[0]}" + time.strftime("%Y-%m-%d_%Hh%M")
        save_folder = join(cfg.dir.exp, cfg.model.method, formatted_pretrain_model, folder_name)

    # logging file
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %I:%M:%S %p')

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # check if there is a previous config file
    if isfile(join(save_folder, 'config.yaml')):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(join(save_folder, 'config.yaml')))

        for override_arg in override_args:
            key, value = override_arg.split('=')
            OmegaConf.update(cfg, key, value)
            logger.info(f"\tOverriding config: {key} = {value}")

    cfg.save_folder = save_folder

    if run_mode == 'train':
        os.makedirs(cfg.dir.exp, exist_ok=True)
        os.makedirs(cfg.save_folder, exist_ok=True)
        os.makedirs(join(cfg.save_folder, 'checkpoints'), exist_ok=True)

    # viso folder
    if cfg.log.use_viso:
        os.makedirs(join(cfg.save_folder, 'viso'), exist_ok=True)


    # select config

    if cfg.model.inference.num_workers is None:
        cfg.model.inference.num_workers = cfg.model.train.num_workers

    return cfg




def formatted_time(elapsed_time):
    days = int(elapsed_time / (24 * 3600))
    hours = int((elapsed_time % (24 * 3600)) / 3600)
    minutes = int((elapsed_time % 3600) / 60)
    seconds = int(elapsed_time % 60)

    time_str = ""

    if days > 0:
        time_str += f"{days} day(s), "
    if hours > 0:
        time_str += f"{hours} hour(s), "
    if minutes > 0:
        time_str += f"{minutes} minute(s), "

    time_str += f"{seconds} second(s)"

    return time_str



def check_gpu_memory(logger, id, prev=None):

    cur_device = torch.cuda.current_device()
    allocated_memory = torch.cuda.memory_allocated() / 1024 / 1024

    if prev is None:
        prev = allocated_memory

    logger.info(f"ID {id} Device {cur_device} Memory allocated {allocated_memory:.3f} MB increment {allocated_memory - prev:.3f} MB max peak {torch.cuda.max_memory_allocated() / 1024 / 1024:.3f}")

    return allocated_memory


# -------------------------------------------------------------------------------------------
# Select Prediction Arranger
# -------------------------------------------------------------------------------------------
def pred_arranger(data_dict, batch_predictions):
    for prediction in batch_predictions:
        item = {}
        item['url'] = prediction[0]
        item['keyphrases'] = [keyphrase.split() for keyphrase in prediction[1]]
        if len(prediction) > 2:
            item['scores'] = prediction[2]

        data_dict[item['url']] = item


def pred_saver(cfg, tot_predictions, filename):
    with open(filename, 'w', encoding='utf-8') as f_pred:
        for url, item in tot_predictions.items():
            data = {}
            data['url'] = url
            data['keyphrases'] = item['keyphrases']
            if "scores" in item:
                data['scores'] = item['scores']
            f_pred.write("{}\n".format(json.dumps(data)))
        f_pred.close()
    logger.info('Success save %s prediction file' % filename)


# -------------------------------------------------------------------------------------------
# Common Functions
# -------------------------------------------------------------------------------------------
def set_seed(cfg):
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.distributed:
            torch.cuda.manual_seed_all(cfg.seed)


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if not math.isnan(val):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
        else:
            logger.warning("\nNaN value in current batch loss")


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total