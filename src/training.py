"""
Main script for training and testing the models.
"""

import hydra
import torch
import wandb
from tqdm import tqdm

from utils import init_wandb, set_deterministic, get_dataloaders, get_device
from omegaconf import OmegaConf
from accelerate import Accelerator


def epoch_train(
        model, loader, optimizer, criterion, device, accelerator, scheduler=None
):
    epoch_loss = 0.0

    model.train()
    for batch in loader:
        optimizer.zero_grad()
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()
        # todo: add scheduler
        # scheduler.step()
        epoch_loss += accelerator.gather(loss.item())

    return epoch_loss.mean().item()


def epoch_evaluate(model, loader, criterion, device, accelerator):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_loss += accelerator.gather(loss.item())

    return epoch_loss.mean().item()


def train(
        device,
        model,
        optimizer,
        criterion,
        accelerator,
        n_epochs,
        train_loader,
        val_loader,
        test_loader
):
    with tqdm(total=n_epochs, desc="Training Progress") as pbar:
        for epoch in range(n_epochs):
            # todo: add scheduler
            train_loss = epoch_train(
                model, train_loader, optimizer, criterion, device, accelerator, scheduler=None
            )
            valid_loss = epoch_evaluate(model, val_loader, criterion, device)
            pbar.set_description(
                f"Train Loss: {train_loss:.3f}"
                f" |  Val. Loss: {valid_loss:.3f} "
            )
            pbar.update(1)
            wandb.log(
                {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss
                }
            )
    test_loss = epoch_evaluate(model, test_loader, criterion, device)
    print(f" Test Loss: {test_loss:.3f} ")
    wandb.log({"test_loss": test_loss})


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg):
    print(f"Hydra configuration:\n{OmegaConf.to_yaml(cfg)}")
    set_deterministic(cfg.seed)
    init_wandb(cfg)  # TODO
    accelerator = Accelerator(
        mixed_precision='no',
        gradient_accumulation_steps=1,
        log_with="wandb",
        # logging_dir="logs" # unexpected argument?
    )
    device = get_device(cfg)
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    print(f"Tokenizer:\n{tokenizer}")
    model = hydra.utils.instantiate(cfg.model)
    model.to(device)
    print(f"Model:\n{model}")
    # todo: freeze model parameters if pretrained
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    print(f"Optimizer:\n{optimizer}")
    train_loader, val_loader, test_loader = get_dataloaders(cfg, tokenizer)
    train_loader, val_loader, test_loader = accelerator.prepare(train_loader, val_loader, test_loader)
    criterion = None  # todo: add criterion
    train(
        device,
        model,
        optimizer,
        criterion,
        accelerator,
        cfg.epochs,
        train_loader,
        val_loader,
        test_loader,
    )


if __name__ == "__main__":
    main()
