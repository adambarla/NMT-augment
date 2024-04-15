import sys
import hydra
import wandb
from omegaconf import OmegaConf
from utils import (
    init_wandb,
    set_deterministic,
    get_dataloaders,
    get_dataset,
    epoch_train,
    epoch_evaluate,
    init_accelerator,
)


def train(
    model,
    optimizer,
    criterion,
    accelerator,
    n_epochs,
    train_loader,
    val_loader,
    test_loader,
    tokenizer_l1,
    tokenizer_l2,
    patience,
):
    min_loss = sys.maxsize
    epochs_since_improvement = 0
    epoch = 0
    for epoch in range(n_epochs):
        print(f"Epoch: {epoch + 1:>{len(str(n_epochs))}d}/{n_epochs}")
        train_loss = epoch_train(
            model,
            train_loader,
            optimizer,
            criterion,
            accelerator,
        )
        valid_loss, valid_bleu = epoch_evaluate(
            model,
            val_loader,
            criterion,
            accelerator,
            tokenizer_l1,
            tokenizer_l2,
        )
        if accelerator.is_main_process:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "valid_bleu": valid_bleu,
                }
            )
        if valid_loss < min_loss:
            min_loss = valid_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        if epochs_since_improvement >= patience:
            break
        print("-" * (len(str(n_epochs)) * 2 + 8))
    if epochs_since_improvement >= patience:
        print(
            f"Early stopping triggered in epoch {epoch + 1}, "
            f"validation loss hasn't improved for {epochs_since_improvement} epochs."
        )
    test_loss, test_bleu = epoch_evaluate(
        model, test_loader, criterion, accelerator, tokenizer_l1, tokenizer_l2
    )
    print(f" Test Loss: {test_loss:.3f} | Test BLEU: {test_bleu:.2f}")
    if accelerator.is_main_process:
        wandb.log({"test_loss": test_loss, "test_bleu": test_bleu})


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg):
    print(f"Hydra configuration:\n{OmegaConf.to_yaml(cfg)}")
    set_deterministic(cfg.seed)
    accelerator = init_accelerator(cfg)
    init_wandb(cfg, accelerator)
    print(f"Accelerator: {accelerator}")
    device = accelerator.device
    print(f"Device: {device}")
    dataset = get_dataset(cfg)
    print(f"Tokenizer {cfg.data.l1}:")
    tokenizer_l1 = hydra.utils.instantiate(
        cfg.tokenizer, dataset=dataset, lang=cfg.data.l1
    )
    print(f"Tokenizer {cfg.data.l2}:")
    tokenizer_l2 = hydra.utils.instantiate(
        cfg.tokenizer, dataset=dataset, lang=cfg.data.l2
    )
    augmenter = None
    if cfg.augmenter is not None:
        augmenter = hydra.utils.instantiate(cfg.augmenter)
    print(f"Augmentation:\n{augmenter or 'No augmentations used.'}")
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg, tokenizer_l1, tokenizer_l2, augmenter, dataset
    )
    assert tokenizer_l1.pad_token_id == tokenizer_l2.pad_token_id
    assert tokenizer_l1.bos_token_id == tokenizer_l2.bos_token_id
    assert tokenizer_l1.eos_token_id == tokenizer_l2.eos_token_id
    model = hydra.utils.instantiate(
        cfg.model,
        device=device,
        pad_token_id=tokenizer_l1.pad_token_id,
        bos_token_id=tokenizer_l1.bos_token_id,
        eos_token_id=tokenizer_l1.eos_token_id,
    )
    model.to(accelerator.device)
    print(f"Model:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    print(f"Optimizer:\n{optimizer}\n")
    criterion = hydra.utils.instantiate(cfg.criterion)
    train_loader, val_loader, test_loader, model, optimizer = accelerator.prepare(
        train_loader, val_loader, test_loader, model, optimizer
    )
    train(
        model,
        optimizer,
        criterion,
        accelerator,
        cfg.epochs,
        train_loader,
        val_loader,
        test_loader,
        tokenizer_l1,
        tokenizer_l2,
        patience=cfg.patience,
    )


if __name__ == "__main__":
    main()
