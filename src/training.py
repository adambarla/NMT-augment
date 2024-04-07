"""
Main script for training and testing the models.
"""

import hydra
import torch
import wandb
from tqdm import tqdm

from utils import (
    init_wandb,
    set_deterministic,
    get_dataloaders,
    get_device,
    get_dataset,
)
from omegaconf import OmegaConf
from accelerate import Accelerator
from sacrebleu import corpus_bleu

def epoch_train(model, loader, optimizer, criterion, device, accelerator):
    epoch_loss = 0.0

    model.train()
    with tqdm(total=len(loader), desc="Training Progress") as pbar:
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.transpose(0, 1).to(device)
            targets = targets.transpose(0, 1).to(device)
            outputs = model(inputs, targets[:-1, :])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]), targets[1:, :].reshape(-1)
            )
            accelerator.backward(loss)
            optimizer.step()
            epoch_loss += accelerator.gather(loss.item())
            pbar.set_description(f"Train Loss: {epoch_loss / (i + 1.0):.3f}")
            pbar.update(1)
    return epoch_loss / len(loader)


def epoch_evaluate(model, loader, criterion, device, accelerator, tokenizer):
    epoch_loss = 0.0
    model.eval()

    hypotheses = []
    references = []

    inputs = targets = None
    with torch.no_grad():
        with tqdm(total=len(loader), desc="Training Progress") as pbar:
            for i, batch in enumerate(loader):
                inputs, targets = batch
                inputs = inputs.transpose(0, 1).to(device)
                targets = targets.transpose(0, 1).to(device)
                outputs = model(inputs, targets[:-1, :])
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[-1]), targets[1:, :].reshape(-1)
                )
                epoch_loss += accelerator.gather(loss.item())
                pbar.set_description(f"Valid Loss: {epoch_loss / (i + 1.0):.3f}")
                pbar.update(1)

                # Generate translations
                generated_seqs = model.translate(inputs, max_length=targets.size(0))
                generated_texts = [tokenizer.decode(seq.tolist()) for seq in generated_seqs.transpose(0, 1)]
                target_texts = [tokenizer.decode(seq) for seq in targets.transpose(0, 1)]

                hypotheses.extend(each for each in generated_texts)
                references.extend(ref for ref in target_texts)



        translations = model.translate(
            inputs, max_length=inputs.shape[0] * 1.05, context_size=inputs.shape[0]
        )
        for i in range(3):
            print(
                f"\n\t input: {tokenizer.decode(inputs[:,i])[0]}\n"
                f"\ttarget: {tokenizer.decode(targets[:,i])[0]}\n"
                f"\toutput: {tokenizer.decode(translations[:,i])[0]}"
            )


    # Calculate BLEU score
    new_hypotheses = []
    for i in range(len(hypotheses)):
        new_hypotheses.append(hypotheses[i][0])
    bleu_score = corpus_bleu(new_hypotheses, references).score
    print(f"BLEU score: {bleu_score:.2f}")

    return epoch_loss / len(loader), bleu_score


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
    tokenizer,
):
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
            model, val_loader, criterion, device, accelerator, tokenizer
        )
        wandb.log({"train_loss": train_loss, "valid_loss": valid_loss, "valid_bleu": valid_bleu})
        print("\n" + "-" * (len(str(n_epochs)) * 2 + 8))
    test_loss, test_bleu = epoch_evaluate(
        model, test_loader, criterion, device, accelerator, tokenizer
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
    tokenizer = hydra.utils.instantiate(cfg.tokenizer, dataset=dataset)
    cfg.src_vocab_size = tokenizer.vocab_size
    cfg.tgt_vocab_size = tokenizer.vocab_size
    print(f"Tokenizer:\n{tokenizer}")


    train_loader, val_loader, test_loader = get_dataloaders(cfg, tokenizer, dataset)
    train_loader, val_loader, test_loader = accelerator.prepare(
        train_loader, val_loader, test_loader
    )
    model = hydra.utils.instantiate(
        cfg.model,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model.to(device)
    print(f"Model:\n{model}")
    # todo: freeze model parameters if pretrained
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    print(f"Optimizer:\n{optimizer}")

    criterion = hydra.utils.instantiate(cfg.criterion)
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
        tokenizer,
    )


if __name__ == "__main__":
    main()
