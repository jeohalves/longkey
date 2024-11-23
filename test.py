import logging
import os
import sys
from functools import partial
from os.path import join
from omegaconf import DictConfig, OmegaConf
import hydra
import copy

from longkey import utils
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from longkey import dataloader, networks, KeyphraseExtraction

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoTokenizer
from longkey.evaluator import select_eval_script
from hydra.core.hydra_config import HydraConfig

logger = logging.getLogger()


def build_test_dataloader(cfg, tokenizer, MethodDataloaderClass, hidden_size):
    """
    Build the test dataloader for the model. This function is called by the main worker function to create the dataloader for testing the model. It uses the MethodDataloaderClass to create the dataset and dataloader for testing. It also handles distributed data loading if necessary.

    Args:
        cfg (DictConfig): Configuration file. Must contain all necessary parameters for the model.
        tokenizer (AutoTokenizer): Tokenizer for the model using the HuggingFace Transformers library.
        MethodDataloaderClass (class): Class for the keyphrase extractor dataloader. Must be a subclass of longkey.dataloader.AbstractData.
        hidden_size (int): Hidden size of the model.

    Returns:
        test_dataset (Dataset): Dataset for testing the model.
        test_data_loader (DataLoader): Dataloader for testing the model.
        test_sampler (Sampler): Sampler for testing the model.
    """

    test_dataset = MethodDataloaderClass(
        **{
            "cfg": cfg,
            "tokenizer": tokenizer,
            "max_token": cfg.model.inference.max_token,
            "mode": "test",
        }
    )

    test_sampler = (
        torch.utils.data.sampler.SequentialSampler(test_dataset)
        if not cfg.distributed
        else DistributedSampler(test_dataset)
    )
    batchify_features_for_test = partial(
        MethodDataloaderClass.batchify_test,
        encoder_output_dim=hidden_size,
        global_attention=cfg.model.global_attention,
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=cfg.model.inference.batch_size,
        sampler=test_sampler,
        num_workers=cfg.model.inference.num_workers,
        collate_fn=batchify_features_for_test,
        pin_memory=cfg.runtime.pin_memory,
    )

    return test_dataset, test_data_loader, test_sampler


# -------------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="longkey/configs/", config_name="base")
def main(cfg: DictConfig) -> None:
    """
    Main function for testing the model. This function is called by Hydra. It initializes the model, tokenizer, and dataloader, and runs the evaluation script. It also handles distributed testing if necessary.

    Args:
        cfg (DictConfig): Configuration file. Must contain all necessary parameters for the model.
    """
    override_args = HydraConfig.get().job.override_dirname.split(',')
    cfg = utils.init_cfg_config(cfg, "test", override_args)
    logger.info(OmegaConf.to_yaml(cfg))

    if cfg.distributed:
        ngpus_per_node = torch.cuda.device_count()
        logger.info(f"Found total gpus = {ngpus_per_node}")
        cfg.world_size = ngpus_per_node * cfg.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(cfg, ngpus_per_node))

    else:
        if cfg.device is None:
            cfg.device = 0
        main_worker(device=cfg.device, cfg=cfg)


def main_worker(device, cfg, ngpus_per_node=None):
    """
    Main worker function for distributed and single GPU inference with PyTorch.

    Args:
        device (int): Current device to run the model on. If distributed, this is the local rank. If not, this is the GPU index (or 0 if no GPU).
        cfg (DictConfig): Configuration file. Must contain all necessary parameters for the model.
        ngpus_per_node (int, optional): Number of GPUS per node. Only used in distributed inference. Defaults to None.

    """

    cfg = copy.deepcopy(cfg)

    if cfg.distributed and device <= 0:
        logger.setLevel(logging.INFO)  # logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter("[%(asctime)s] %(message)s ", "%Y/%m/%d %I:%M:%S %p")

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

        log_file = join(cfg.save_folder, "logging.txt")

        logfile = logging.FileHandler(log_file, "w")
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
        logger.info("COMMAND: %s" % " ".join(sys.argv))
        logger.info("preprocess_folder = {}".format(cfg.dir.data))
        logger.info("Pretrain Model Type = {}".format(cfg.model.pretrain_model))

    # disable logging for processes except 0 on every node
    if device > 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f

    if cfg.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)

    if cfg.distributed:
        rank = cfg.rank * ngpus_per_node + device
        dist.init_process_group(
            backend=cfg.dist_backend,
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=rank,
        )

    # Setting device
    if cfg.device >= 0:
        torch.cuda.set_device(device)
    else:
        device = "cpu"
        cfg.runtime.cuda = False

    cfg.device = device

    # -------------------------------------------------------------------------------------------
    # Setup CUDA, GPU & distributed training
    if not torch.cuda.is_available():
        cfg.runtime.cuda = False
    torch.backends.cudnn.benchmark = cfg.runtime.cuda and cfg.runtime.benchmark

    logger.info(
        "Process rank: %s, device: %s, distributed testing: %s",
        cfg.rank,
        torch.device(device),
        cfg.distributed,
    )

    # -------------------------------------------------------------------------------------------
    utils.set_seed(cfg)

    # -------------------------------------------------------------------------------------------
    # init tokenizer & Converter
    logger.info(
        "start setting tokenizer, dataset and dataloader (local_rank = {})... ".format(
            cfg.rank
        )
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrain_model)
    except EnvironmentError:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # -------------------------------------------------------------------------------------------
    # Select dataloader
    _, num_labels = networks.get_class(cfg)

    model_config = AutoConfig.from_pretrained(
        cfg.model.pretrain_model, num_labels=num_labels
    )

    MethodDataloaderClass = dataloader.get_class(cfg.model.method)

    # -------------------------------------------------------------------------------------------

    test_dataset, test_data_loader, _ = build_test_dataloader(
        cfg, tokenizer, MethodDataloaderClass, model_config.hidden_size
    )

    logger.info("Successfully Preprocess test Features !")

    # -------------------------------------------------------------------------------------------
    # Prepare Model & Optimizer
    # -------------------------------------------------------------------------------------------
    if cfg.distributed and cfg.rank != 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info(
        " ************************** Initialize Model & Optimizer ************************** "
    )

    if cfg.load_checkpoint is not None:
        relative_path = join(cfg.save_folder, 'checkpoints', cfg.load_checkpoint)

        if os.path.isfile(relative_path):
            cfg.load_checkpoint = relative_path

        if os.path.isfile(cfg.load_checkpoint):
            model = KeyphraseExtraction.load_checkpoint(cfg.load_checkpoint, cfg)[0]
        else:
            raise ValueError(
                f"Invalid checkpoint for evaluation {cfg.load_checkpoint} {os.path.isfile(cfg.load_checkpoint)}"
            )
    else:
        raise ValueError(
            f"No checkpoint was provided for evaluation {cfg.load_checkpoint}"
        )

    # -------------------------------------------------------------------------------------------
    if cfg.distributed and cfg.rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    # -------------------------------------------------------------------------------------------

    # set model device
    model.set_device()

    if cfg.distributed:
        model.distribute(torch.device(model.cfg.device))

    logger.info("Testing parameters %s", cfg)
    logger.info(
        " ************************** Running inference ************************** "
    )

    # -------------------------------------------------------------------------------------------
    # Method Select
    evaluate_script, main_metric_name = select_eval_script()

    stats = {"timer": utils.Timer(), "epoch": 0, main_metric_name: 0}
    model.scaler = torch.amp.GradScaler("cuda")

    test_candidate = MethodDataloaderClass.decoder(
        cfg,
        test_data_loader,
        test_dataset,
        model,
        MethodDataloaderClass.test_input_refactor,
        "test",
    )

    if cfg.distributed:
        all_test_candidate = [dict() for i in range(torch.cuda.device_count())]
        torch.distributed.barrier()
        torch.distributed.all_gather_object(all_test_candidate, test_candidate)
        if cfg.rank != 0:
            torch.distributed.barrier()  # BEGIN Make sure only the first process in distributed training evaluates
        else:
            test_candidate = dict()
            for dict_rank in all_test_candidate:
                test_candidate.update(dict_rank)
            torch.distributed.barrier()  # END Make sure only the first process in distributed training evaluates

    if cfg.rank == 0:
        stats = evaluate_script(
            cfg, test_candidate, stats, mode="test", metric_name=main_metric_name
        )

    if cfg.distributed:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
