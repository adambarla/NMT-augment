import sys
import hydra
import wandb
from utils import (
    init_wandb,
    set_deterministic,
    get_dataloaders,
    get_device,
    get_dataset,
    epoch_train,
    epoch_evaluate,
)
from omegaconf import OmegaConf
from accelerate import Accelerator


def train(
    device,
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
            device,
            accelerator,
        )
        valid_loss, valid_bleu = epoch_evaluate(
            model,
            val_loader,
            criterion,
            device,
            accelerator,
            tokenizer_l1,
            tokenizer_l2,
        )
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
        model, test_loader, criterion, device, accelerator, tokenizer_l1, tokenizer_l2
    )
    print(f" Test Loss: {test_loss:.3f} | Test BLEU: {test_bleu:.2f}")
    wandb.log({"test_loss": test_loss, "test_bleu": test_bleu})


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg):
    print(f"Hydra configuration:\n{OmegaConf.to_yaml(cfg)}")
    set_deterministic(cfg.seed)
    init_wandb(cfg)  # TODO
    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=1,
        log_with="wandb",
        # logging_dir="logs" # unexpected argument?
    )
    device = get_device(cfg)
    dataset = get_dataset(cfg)
    tokenizer_l1 = hydra.utils.instantiate(
        cfg.tokenizer, dataset=dataset, lang=cfg.data.l1
    )
    print(f"Tokenizer {cfg.data.l1}:\n{tokenizer_l1}")
    tokenizer_l2 = hydra.utils.instantiate(
        cfg.tokenizer, dataset=dataset, lang=cfg.data.l2
    )
    print(f"Tokenizer {cfg.data.l2}:\n{tokenizer_l2}")
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg, tokenizer_l1, tokenizer_l2, dataset
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
    #model.to(device)
    print(f"Model:\n{model}")
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    print(f"Optimizer:\n{optimizer}")
    criterion = hydra.utils.instantiate(cfg.criterion)

    #add accelerator
    train_loader, val_loader, test_loader, model, optimizer = accelerator.prepare(
        train_loader, val_loader, test_loader, model, optimizer)

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
        tokenizer_l1,
        tokenizer_l2,
        patience=cfg.patience,
    )


if __name__ == "__main__":
    main()
