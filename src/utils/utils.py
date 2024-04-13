import wandb
from accelerate import Accelerator
from omegaconf import OmegaConf
from datetime import datetime
import torch


def get_device(cfg):
    if cfg.device is not None:
        return torch.device(cfg.device)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    return device

def init_accelerator(cfg):
    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=1,
    )
    print(f"object cfg.wandb: {OmegaConf.to_object(cfg.wandb)}")
    return accelerator

def init_wandb(cfg, accelerator):
    if accelerator.is_local_main_process:
        if cfg.group is None:
            g = cfg.model._target_.split(".")[-1]  # group of the run determined by model
            cfg.group = f"{g}"
        if cfg.name is None:
            t = datetime.now().strftime("%Y%m%d_%H%M%S")
            # add other parameters of name if needed
            tok = cfg.tokenizer._target_.split(".")[-1]
            l1 = cfg.data.l1
            l2 = cfg.data.l2
            cfg.name = f"{tok}_{l1}_{l2}_{t}"

        wandb.init(
            name=cfg.name,
            group=cfg.group,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            )
