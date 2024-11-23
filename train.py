import os
import sys
import torch
import logging
import traceback
from tqdm import tqdm
from os.path import isfile, join

from omegaconf import DictConfig, OmegaConf
import hydra
import copy

from longkey import utils, dataloader, networks, KeyphraseExtraction
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from tensorboardX import SummaryWriter
import random
from functools import partial
from longkey.evaluator import select_eval_script
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig

logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------------------------------


def train(cfg, data_loader_dict, model, train_input_refactor, stats, writer):
    """
    Train the model for one epoch. It iterates through the training data loader and updates the model. It also logs the training loss and other statistics. The training loss is reset every `cfg.log.display_iter` iterations and the individual losses are reset every iteration.


    Args:
        cfg (DictConfig): Configuration file. Must contain all necessary parameters for the model.
        data_loader_dict (dict): Data loader dictionary containing the training data loader, iterator, and length.
        model (KeyphraseExtraction): An instance of the KeyphraseExtraction class. The keyphrase extraction method with the language model must be loaded into this instance.
        train_input_refactor (_type_):
        stats (dict): Dictionary containing the timer and other statistics.
        writer (_type_): _description_
    """
    logger.info(
        f"start training {cfg.model.method} ({stats['epoch']} epoch) || local_rank = {cfg.rank}..."
    )

    train_loss = utils.AverageMeter()
    individual_loss = None
    epoch_time = utils.Timer()

    epoch_loss = 0
    epoch_step = 1e-8

    epoch_len = (
        cfg.model.train.max_steps_per_epoch
        if cfg.model.train.max_steps_per_epoch
        and cfg.model.train.max_steps_per_epoch < data_loader_dict["length"]
        else data_loader_dict["length"]
    )

    pbar = tqdm(range(epoch_len), desc="Train_Iteration", disable=cfg.rank != 0)

    for step in pbar:
        try:
            try:
                inputs, _ = train_input_refactor(
                    next(data_loader_dict["loader_iter"]), model.cfg.device
                )
            except StopIteration:
                data_loader_dict["loader_iter"] = iter(
                    data_loader_dict["train_data_loader"]
                )
                inputs, _ = train_input_refactor(
                    next(data_loader_dict["loader_iter"]), model.cfg.device
                )
            loss = model.update(step, inputs)
        except:  # noqa: E722
            logging.error(str(traceback.format_exc()))
            torch.cuda.empty_cache()
            raise ValueError("training exception")

        if isinstance(loss, dict):
            if individual_loss is None:
                individual_loss = dict()
                for key in loss.keys():
                    individual_loss[key] = utils.AverageMeter()
            else:
                for key in loss.keys():
                    individual_loss[key].update(loss[key])

            loss = sum(loss.values())
            train_loss.update(loss)

        else:
            train_loss.update(loss)

        epoch_loss += loss
        epoch_step += 1

        pbar.set_description(
            f"Training... epoch loss {(epoch_loss / epoch_step):.6f} | loss {train_loss.avg:.6f}"
        )

        if step > 0 and step % cfg.log.display_iter == 0:
            if cfg.rank == 0 and cfg.log.use_viso:
                writer.add_scalar("train/loss", train_loss.avg, model.updates)
                writer.add_scalar(
                    "train/lr", model.scheduler.get_last_lr()[0], model.updates
                )

            if individual_loss is None:
                individual_losses = "|"
            else:
                individual_losses = ", ".join(
                    [
                        f"{key}: {individual_loss[key].avg:.6f}"
                        for key in individual_loss.keys()
                    ]
                )
                individual_losses = f"| {individual_losses} |"

            logging.info(
                f"TRAIN: Epoch = {stats['epoch']} | iter = {step}/{epoch_len} | epoch loss = {(epoch_loss / epoch_step):.6f} | loss = {train_loss.avg:.6f} | lr = {model.scheduler.get_last_lr()[0]:.3e} {individual_losses} {model.updates} updates | elapsed time = {utils.formatted_time(stats['timer'].time())}"
            )

            train_loss.reset()
            if individual_loss is not None:
                for key in individual_loss.keys():
                    individual_loss[key].reset()

            torch.cuda.empty_cache()

    logging.info(
        "Epoch Mean Loss = %.8f ( Epoch = %d ) | Time for epoch = %s \n"
        % (
            (epoch_loss / epoch_step),
            stats["epoch"],
            utils.formatted_time(epoch_time.time()),
        )
    )


def build_train_dataloader(cfg, tokenizer, MethodDataloaderClass, hidden_size):
    # build train dataloader
    train_dataset = MethodDataloaderClass(
        **{
            "cfg": cfg,
            "tokenizer": tokenizer,
            "max_token": cfg.model.train.max_token,
            "mode": "train",
        }
    )

    train_sampler = (
        torch.utils.data.sampler.RandomSampler(train_dataset)
        if not cfg.distributed
        else DistributedSampler(train_dataset, shuffle=True)
    )
    batchify_features_for_train = partial(
        MethodDataloaderClass.batchify_train,
        encoder_output_dim=hidden_size,
        global_attention=cfg.model.global_attention,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=cfg.model.train.batch_size,
        sampler=train_sampler,
        num_workers=cfg.model.train.num_workers,
        collate_fn=batchify_features_for_train,
        pin_memory=cfg.runtime.pin_memory,
    )

    return train_dataset, train_data_loader, train_sampler


def build_dev_dataloader(cfg, tokenizer, MethodDataloaderClass, hidden_size):
    # build dev dataloader
    dev_dataset = MethodDataloaderClass(
        **{
            "cfg": cfg,
            "tokenizer": tokenizer,
            "max_token": cfg.model.inference.max_token,
            "mode": "dev",
        }
    )

    dev_sampler = (
        torch.utils.data.sampler.SequentialSampler(dev_dataset)
        if not cfg.distributed
        else DistributedSampler(dev_dataset)
    )
    batchify_features_for_test = partial(
        MethodDataloaderClass.batchify_test,
        encoder_output_dim=hidden_size,
        global_attention=cfg.model.global_attention,
    )

    dev_data_loader = DataLoader(
        dev_dataset,
        batch_size=cfg.model.inference.batch_size,
        sampler=dev_sampler,
        num_workers=cfg.model.inference.num_workers,
        collate_fn=batchify_features_for_test,
        pin_memory=cfg.runtime.pin_memory,
    )

    return dev_dataset, dev_data_loader, dev_sampler


# -------------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="longkey/configs/", config_name="base")
def main(cfg: DictConfig) -> None:
    """
    Main function for training a keyphrase extraction model. It initializes the model, tokenizer, and data loaders, and then trains the model for the specified number of epochs. It also evaluates the model on the dev set after every `cfg.model.val_epochs_to_skip` epochs. The best model is saved based on the evaluation metric.

    Args:
        cfg (DictConfig): Configuration file. Must contain all necessary parameters for the model. The configuration file is generated by Hydra based on the configuration files in the `longkey/configs/` directory.
    """

    override_args = HydraConfig.get().job.override_dirname.split(',')
    cfg = utils.init_cfg_config(cfg, "train", override_args)
    logger.info(OmegaConf.to_yaml(cfg))

    if cfg.distributed:
        ngpus_per_node = torch.cuda.device_count()
        logger.info(f"Found total gpus = {ngpus_per_node}")
        world_size = ngpus_per_node * cfg.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(cfg, ngpus_per_node, world_size))

    else:
        if cfg.device is None:
            cfg.device = 0
        main_worker(device=cfg.device, cfg=cfg)


def main_worker(device, cfg, ngpus_per_node=None, world_size=1):
    """
    Main worker function for distributed and single GPU training with PyTorch.

    Args:
        device (int): Current device to run the model on. If distributed, this is the local rank. If not, this is the GPU index (or 0 if no GPU).
        cfg (DictConfig): Configuration file. Must contain all necessary parameters for the model.
        ngpus_per_node (int, optional): Number of GPUS per node. Only used in distributed training. Defaults to None.

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

    # -------------------------------------------------------------------------------------------

    # disable logging for processes except 0 on every node
    if device > 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f

    if cfg.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)

    if cfg.distributed:
        cfg.rank = cfg.rank * ngpus_per_node + device
        dist.init_process_group(
            backend=cfg.dist_backend,
            init_method=cfg.dist_url,
            world_size=world_size,
            rank=cfg.rank,
        )

    # # Setting device
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
        "Process rank: %s, device: %s, distributed training: %s, 16-bits training: %s",
        cfg.rank,
        torch.device(device),
        cfg.distributed,
        cfg.mixed,
    )

    # -------------------------------------------------------------------------------------------
    utils.set_seed(cfg)
    # Make sure only the first process in distributed training will download model & vocab

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

    train_dataset, train_data_loader, train_sampler = build_train_dataloader(
        cfg, tokenizer, MethodDataloaderClass, model_config.hidden_size
    )
    logger.info("Successfully Preprocess Training Features !")

    # -------------------------------------------------------------------------------------------

    dev_dataset, dev_data_loader, dev_sampler = build_dev_dataloader(
        cfg, tokenizer, MethodDataloaderClass, model_config.hidden_size
    )

    logger.info("Successfully Preprocess Dev Features !")

    # -------------------------------------------------------------------------------------------

    len_train_data_loader = (
        len(train_data_loader)
        if cfg.model.train.max_steps_per_epoch is None
        else cfg.model.train.max_steps_per_epoch
    )
    updates_per_epoch = len_train_data_loader // cfg.model.gradient_accumulation_steps

    # -------------------------------------------------------------------------------------------
    # Prepare Model & Optimizer
    # -------------------------------------------------------------------------------------------
    if cfg.distributed and cfg.rank != 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info(
        " ************************** Initialize Model & Optimizer ************************** "
    )

    last_epoch = -1
    optimizer_state_dict = None
    scheduler_state_dict = None

    if cfg.load_checkpoint is not None:
        relative_path = join(cfg.save_folder, 'checkpoints', cfg.load_checkpoint)

        if os.path.isfile(relative_path):
            cfg.load_checkpoint = relative_path

        if os.path.isfile(cfg.load_checkpoint):
            (
                model,
                last_epoch,
                optimizer_state_dict,
                scheduler_state_dict,
            ) = KeyphraseExtraction.load_checkpoint(cfg.load_checkpoint, cfg)
        else:
            raise ValueError(
                f"Invalid checkpoint to resume training {cfg.load_checkpoint}"
            )
    else:
        logger.info("Training model from scratch...")
        model = KeyphraseExtraction(cfg)

    # initial optimizer
    model.init_optimizer(
        optimizer_state_dict,
        scheduler_state_dict,
        updates_per_epoch,
        cfg.model.max_train_epochs,
        last_epoch,
    )

    # -------------------------------------------------------------------------------------------
    if cfg.distributed and cfg.rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    # -------------------------------------------------------------------------------------------

    # set model device
    model.set_device()

    if cfg.distributed:
        model.distribute(torch.device(device))

    if cfg.rank == 0 and cfg.log.use_viso:
        tb_writer = SummaryWriter(join(cfg.save_folder, "viso"))
    else:
        tb_writer = None

    logger.info("Training/evaluation parameters %s", cfg)
    logger.info(
        " ************************** Running training ************************** "
    )
    logger.info("  Num Train examples = %d", len(train_dataset))
    logger.info("  Num Train Epochs = %d", cfg.model.max_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", cfg.model.train.batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        cfg.model.train.batch_size
        * cfg.model.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if cfg.distributed else 1),
    )
    logger.info(
        "  Gradient Accumulation steps = %d", cfg.model.gradient_accumulation_steps
    )
    logger.info(
        " *********************************************************************** "
    )

    # -------------------------------------------------------------------------------------------
    # Method Select
    evaluate_script, main_metric_name = select_eval_script()

    # -------------------------------------------------------------------------------------------
    # start training
    # -------------------------------------------------------------------------------------------
    model.zero_grad()
    prev_checkpoint_name = None
    stats = {"timer": utils.Timer(), "epoch": 0, main_metric_name: 0}
    model.scaler = torch.amp.GradScaler("cuda")

    model.optimizer.step()

    if last_epoch == -1:
        last_epoch = 0
    sampler_epochs = list(range(0, (cfg.model.max_train_epochs + 2)))
    random.shuffle(sampler_epochs)

    train_data_loader_dict = {
        "train_data_loader": train_data_loader,
        "loader_iter": iter(train_data_loader),
        "length": len(train_data_loader),
    }

    checkpoint_folder = join(cfg.save_folder, "checkpoints")

    for epoch in range(last_epoch + 1, (cfg.model.max_train_epochs + 1)):
        stats["epoch"] = epoch
        if cfg.distributed:
            train_sampler.set_epoch(sampler_epochs[epoch])

        # train
        train(
            cfg,
            train_data_loader_dict,
            model,
            MethodDataloaderClass.train_input_refactor,
            stats,
            tb_writer,
        )

        # save last checkpoint
        if cfg.rank == 0:
            checkpoint_name = (
                "final.tar"
                if epoch == cfg.model.max_train_epochs
                else f"epoch_{epoch}.tar"
            )

            cfg.load_checkpoint = checkpoint_name

            # save config
            with open(join(cfg.save_folder, "config.yaml"), "w") as f:
                saved_cfg = copy.deepcopy(cfg)
                # reset computer dependent path
                saved_cfg.save_folder = saved_cfg.dir.exp = saved_cfg.dir.data = "???"

                OmegaConf.save(saved_cfg, f)
                logger.info("config.yaml saved.")

            # save checkpoint
            model.save_checkpoint(
                join(checkpoint_folder, checkpoint_name),
                stats["epoch"],
                final=epoch == cfg.model.max_train_epochs,
            )

            # remove previous checkpoint
            if prev_checkpoint_name and isfile(
                join(checkpoint_folder, prev_checkpoint_name)
            ):
                try:
                    os.remove(join(checkpoint_folder, prev_checkpoint_name))
                    logger.info(
                        "previous checkpoint {} removed.".format(prev_checkpoint_name)
                    )
                except BaseException:
                    logger.warning("WARN: removing failed... continuing anyway.")

            prev_checkpoint_name = checkpoint_name

        if 1 < epoch < cfg.model.max_train_epochs and (
            epoch % cfg.model.val_epochs_to_skip != 0
        ):
            logger.info("SKIPPING validation in this epoch")
            continue

        # previous metric score
        prev_metric_score = stats[main_metric_name]

        dev_candidate = MethodDataloaderClass.decoder(
            cfg,
            dev_data_loader,
            dev_dataset,
            model,
            MethodDataloaderClass.test_input_refactor,
            "dev",
        )

        if cfg.distributed:
            all_dev_candidate = [dict() for i in range(torch.cuda.device_count())]
            torch.distributed.barrier()
            torch.distributed.all_gather_object(all_dev_candidate, dev_candidate)
            if cfg.rank != 0:
                torch.distributed.barrier()  # BEGIN Make sure only the first process in distributed training evaluates
            else:
                dev_candidate = dict()
                for dict_rank in all_dev_candidate:
                    dev_candidate.update(dict_rank)
                torch.distributed.barrier()  # END Make sure only the first process in distributed training evaluates

        if cfg.rank == 0:
            stats = evaluate_script(
                cfg, dev_candidate, stats, mode="dev", metric_name=main_metric_name
            )
            # new metric score
            new_metric_score = stats[main_metric_name]

            # save best checkpoint
            if new_metric_score > prev_metric_score:
                logger.info("-" * 60)
                logger.info(
                    f"UPDATE! Epoch = {stats['epoch']} | {main_metric_name} = {new_metric_score:.4f} | previous {main_metric_name} = {prev_metric_score:.4f}"
                )
                logger.info("-" * 60)

                checkpoint_name = "best.tar"
                model.save_checkpoint(
                    join(checkpoint_folder, checkpoint_name), stats["epoch"], final=True
                )

        if cfg.distributed:
            torch.distributed.barrier()

    if cfg.rank == 0:
        tb_writer.close()


if __name__ == "__main__":
    main()
