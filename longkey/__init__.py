from . import dataloader
from . import networks
from . import evaluator
from . import utils

from .constant import (
    PAD,
    UNK,
    BOS,
    EOS,
    DIGIT,
    PAD_WORD,
    UNK_WORD,
    BOS_WORD,
    EOS_WORD,
    DIGIT_WORD,
    Idx2Tag,
    Tag2Idx,
    IdxTag_Converter,
)

import logging
import torch
from transformers import AutoConfig, AutoModel

logger = logging.getLogger()


class KeyphraseExtraction(object):
    def __init__(self, cfg, state_dict=None):
        self.cfg = cfg
        self.updates = 0

        # select model
        network, num_labels = networks.get_class(cfg)

        model_config = AutoConfig.from_pretrained(
            cfg.model.pretrain_model, num_labels=num_labels
        )

        # Custom configs
        model_config.hidden_dropout_prob = cfg.model.hidden_dropout_prob
        model_config.max_encoder_token_size = (
            cfg.model.train.max_chunk_token
            if cfg.model.train.max_chunk_token > 0
            else cfg.model.train.max_token
        )
        model_config.max_phrase_words = cfg.model.max_phrase_words
        model_config.output_hidden_states = True
        model_config.use_checkpoint = cfg.runtime.use_checkpoint

        # load pretrained model
        if cfg.rank == 0:
            logger.info(model_config)

        self.network = network(model_config)

        model = (
            AutoModel.from_config(model_config)
            .from_pretrained(
                cfg.model.pretrain_model,
                config=model_config,
                ignore_mismatched_sizes=True,
            )
            .to(torch.device(cfg.device))
        )

        if hasattr(model_config, "max_encoder_position_embeddings"):
            self.network.encoder = model.encoder
        else:
            self.network.encoder = model

        # load checkpoint
        if state_dict is not None:
            self.network.load_state_dict(state_dict, strict=False)
            if cfg.rank == 0:
                logger.info("loaded checkpoint state_dict")

    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def init_optimizer(
        self,
        optimizer_state_dict,
        scheduler_state_dict,
        updates_per_epoch,
        max_train_epochs,
        last_epoch=-1,
    ):
        num_total_steps = updates_per_epoch * max_train_epochs
        param_optimizer = list(self.network.named_parameters())

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer],
                "weight_decay": self.cfg.optim.weight_decay,
            },
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.cfg.optim.learning_rate
        )

        # load checkpoint
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)
            for param in self.optimizer.state.values():
                if isinstance(param, torch.Tensor):
                    param.data = param.data.to(torch.device(self.cfg.device))
                    if param._grad is not None:
                        param._grad.data = param._grad.data.to(
                            torch.device(self.cfg.device)
                        )
                elif isinstance(param, dict):
                    for subparam in param.values():
                        if isinstance(subparam, torch.Tensor):
                            subparam.data = subparam.data.to(
                                torch.device(self.cfg.device)
                            )
                            if subparam._grad is not None:
                                subparam._grad.data = subparam._grad.data.to(
                                    torch.device(self.cfg.device)
                                )

            logger.info("loaded checkpoint optimizer_state_dict")

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.cfg.optim.learning_rate,
            total_steps=num_total_steps,
            pct_start=self.cfg.optim.warmup_proportion,
            last_epoch=last_epoch,
        )  #

        if scheduler_state_dict is not None:
            self.scheduler.load_state_dict(scheduler_state_dict)
            logger.info("loaded checkpoint scheduler")

    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    # train
    def update(self, step, inputs):
        # Train mode
        self.network.train()

        # run !
        match self.cfg.mixed:
            case "fp16":
                float_type = torch.float16
            case "bf16":
                float_type = torch.bfloat16
            case _:
                float_type = None

        if float_type:
            with torch.autocast(device_type="cuda", dtype=float_type):
                loss = self.network(**inputs)
        else:
            loss = self.network(**inputs)

        if float_type:
            if isinstance(loss, dict):
                self.scaler.scale(sum(loss.values())).backward()
            else:
                self.scaler.scale(loss).backward()
        else:
            if isinstance(loss, dict):
                sum(loss.values()).backward()
            else:
                loss.backward()

        if (step + 1) % self.cfg.model.gradient_accumulation_steps == 0:
            if float_type:
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
            self.updates += 1
            self.scheduler.step()

        if isinstance(loss, dict):
            for key in loss.keys():
                loss[key] = loss[key].cpu().detach().item()
            return loss
        else:
            return loss.cpu().detach().item()

    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def save_checkpoint(self, filename, epoch, final=False):
        network = (
            self.network.module if hasattr(self.network, "module") else self.network
        )
        params = {
            "state_dict": network.state_dict(),
        }

        if final is False:
            params["optimizer"] = self.optimizer.state_dict()
            params["scheduler"] = self.scheduler.state_dict()
            params["epoch"] = epoch

        try:
            torch.save(params, filename)
            logger.info("success save epoch_%d checkpoints !" % epoch)
        except BaseException as e:
            logger.warning(f"WARN: Saving failed... continuing anyway. Exception {e}")

    @staticmethod
    def load_checkpoint(filename, cfg):
        logger.info("Loading model %s" % filename)
        saved_params = torch.load(filename, map_location="cpu", weights_only=False)

        state_dict = saved_params["state_dict"]
        epoch = saved_params["epoch"] if "epoch" in saved_params else 0
        optimizer_state_dict = saved_params["optimizer"] if "optimizer" in saved_params else None
        scheduler_state_dict = saved_params["scheduler"] if "scheduler" in saved_params else None

        model = KeyphraseExtraction(cfg, state_dict)
        logger.info(f"success loaded model weights ! From : {filename}")
        return model, epoch, optimizer_state_dict, scheduler_state_dict

    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def zero_grad(self):
        self.optimizer.zero_grad()
        self.network.zero_grad()

    def set_device(self):
        self.network.to(torch.device(self.cfg.device))

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(
            self.network, find_unused_parameters=self.cfg.runtime.ddp
        )

    def distribute(self, device):
        self.distributed = True
        self.network = torch.nn.parallel.DistributedDataParallel(
            self.network,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=self.cfg.runtime.ddp,
        )
