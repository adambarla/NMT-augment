import wandb
from omegaconf import OmegaConf
from datetime import datetime
import random
import numpy as np
import torch
import os


def get_device(cfg):
    if cfg.device is not None:
        return torch.device(cfg.device)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    return device


def init_wandb(cfg):
    if cfg.group is None:
        g = None  # todo: change automatic selection of the run group
        # g = cfg.model._target_.split(".")[-1]  # group of the run determined by model
        cfg.group = f"{g}"
    if cfg.name is None:
        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        # add other parameters of name if needed
        cfg.name = f"{t}"
    wandb.init(
        name=cfg.name,
        group=cfg.group,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )


def save_checkpoint(model, optimizer, epoch, revision = 0):
    checkpoint_dir = os.path.join("./outputs", "checkpoints", str(revision))
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")