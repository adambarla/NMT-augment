import hydra
import torch
from accelerate.utils import broadcast

import wandb
from omegaconf import OmegaConf
from utils import (
    init_wandb,
    set_deterministic,
    get_loaders,
    get_dataset,
    epoch_train,
    epoch_evaluate,
    init_accelerator,
    init_model,
    init_tokenizers,
    init_augmenter,
    init_metrics,
)


def train(
    model,
    optimizer,
    scheduler,
    criterion,
    accelerator,
    n_epochs,
    train_loader,
    val_loader,
    test_loader,
    tokenizer_l1,
    tokenizer_l2,
    metrics,
    early_stopping,
):
    step = 0
    for epoch in range(n_epochs):
        print(f"Epoch: {epoch + 1:>{len(str(n_epochs))}d}/{n_epochs}")
        if accelerator.is_main_process:
            wandb.log({"epoch": epoch + 1}, step=step)
        epoch_train(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            accelerator,
        )
        step = scheduler.state_dict()["_step_count"]
        val_res = epoch_evaluate(
            model,
            val_loader,
            criterion,
            accelerator,
            tokenizer_l1,
            tokenizer_l2,
            metrics,
            step,
        )
        print("-" * (len(str(n_epochs)) * 2 + 8))
        stop = False
        if accelerator.is_main_process:
            stop = early_stopping.should_stop(val_res)
        stop = broadcast(torch.tensor(stop, dtype=torch.bool, device=accelerator.device))
        if stop.item():
            print(f"Early stopping triggered in epoch {epoch + 1}")
            break
    epoch_evaluate(
        model,
        test_loader,
        criterion,
        accelerator,
        tokenizer_l1,
        tokenizer_l2,
        metrics,
        step,
        name="test",
    )


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg):
    print(f"Hydra configuration:\n{OmegaConf.to_yaml(cfg)}")
    set_deterministic(cfg.seed)
    accelerator = init_accelerator(cfg)
    device = accelerator.device
    init_wandb(cfg, accelerator)
    print(f"Device: {device}")
    dataset = get_dataset(cfg)
    augmenter = init_augmenter(cfg)
    tok_l1, tok_l2 = init_tokenizers(cfg, dataset)
    load_tr, load_va, load_te = get_loaders(cfg, tok_l1, tok_l2, augmenter, dataset)
    model = init_model(cfg, tok_l1, tok_l2, device)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    print(f"Optimizer: {optimizer}")
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    print(f"Scheduler: {scheduler}")
    criterion = hydra.utils.instantiate(cfg.criterion)
    load_tr, load_va, load_te, model, optimizer, scheduler = accelerator.prepare(
        load_tr, load_va, load_te, model, optimizer, scheduler
    )
    metrics = init_metrics(cfg)
    early_stopping = hydra.utils.instantiate(cfg.early_stopping)
    train(
        model,
        optimizer,
        scheduler,
        criterion,
        accelerator,
        cfg.epochs,
        load_tr,
        load_va,
        load_te,
        tok_l1,
        tok_l2,
        metrics,
        early_stopping,
    )


if __name__ == "__main__":
    main()
