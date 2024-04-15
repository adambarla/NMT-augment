import sys
import hydra
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
            scheduler,
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
            print(
                f"Train Loss: {train_loss:.3f} | "
                f"Valid Loss: {valid_loss:.3f} | "
                f"Valid BLEU: {valid_bleu:.2f}"
            )
            wandb.log(
                {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "valid_bleu": valid_bleu,
                }
            )
        print("-" * (len(str(n_epochs)) * 2 + 8))
        if valid_loss < min_loss:
            min_loss = valid_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        if epochs_since_improvement >= patience:
            break
    if epochs_since_improvement >= patience:
        print(
            f"Early stopping triggered in epoch {epoch + 1}, "
            f"validation loss hasn't improved for {epochs_since_improvement} epochs."
        )
    test_loss, test_bleu = epoch_evaluate(
        model,
        test_loader,
        criterion,
        accelerator,
        tokenizer_l1,
        tokenizer_l2,
        name="Test",
    )
    if accelerator.is_main_process:
        print(f" Test Loss: {test_loss:.3f} | Test BLEU: {test_bleu:.2f}")
        wandb.log({"test_loss": test_loss, "test_bleu": test_bleu})


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg):
    print(f"Hydra configuration:\n{OmegaConf.to_yaml(cfg)}")
    set_deterministic(cfg.seed)
    accelerator = init_accelerator(cfg)
    device = accelerator.device
    print(f"Device: {device}")
    init_wandb(cfg, accelerator)
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
        patience=cfg.patience,
    )


if __name__ == "__main__":
    main()
