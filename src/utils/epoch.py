import torch
from sacrebleu import corpus_bleu
from tqdm import tqdm


def epoch_train(model, loader, optimizer, scheduler, criterion, accelerator):
    epoch_loss = 0.0
    model.train()
    with tqdm(total=len(loader), desc="Training Progress") as pbar:
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.transpose(0, 1).to(accelerator.device)
            targets = targets.transpose(0, 1).to(accelerator.device)
            outputs = model(inputs, targets[:-1, :])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]), targets[1:, :].reshape(-1)
            )
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            epoch_loss += accelerator.gather(loss).mean().item()
            pbar.set_description(f"Train Loss: {(epoch_loss / (i + 1.0)):.3f}")
            pbar.update(1)
    return epoch_loss / len(loader)


def epoch_evaluate(
    model,
    loader,
    criterion,
    accelerator,
    tokenizer_l1,
    tokenizer_l2,
    n_examples=3,
    name="Valid",
):
    epoch_loss = 0.0
    model.eval()
    hypotheses = []
    references = []
    inputs = None
    with torch.no_grad():
        with tqdm(total=len(loader), desc=f"{name}") as pbar:
            for i, batch in enumerate(loader):
                inputs, targets = batch
                inputs = inputs.transpose(0, 1).to(accelerator.device)  # L x B
                targets = targets.transpose(0, 1).to(accelerator.device)
                outputs = model(inputs, targets[:-1])  # L x B x V
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[-1]), targets[1:].reshape(-1)
                )
                epoch_loss += accelerator.gather(loss).mean().item()
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    translations = model.module.translate(
                        inputs, buffer=0.5, context_size=inputs.shape[0]
                    )
                else:
                    translations = model.translate(
                        inputs, buffer=0.5, context_size=inputs.shape[0]
                    )
                decoded_translations = tokenizer_l2.decode(translations.transpose(0, 1))
                decoded_targets = tokenizer_l2.decode(targets.transpose(0, 1))
                hypotheses.extend(decoded_translations)
                references.extend(decoded_targets)
                pbar.set_description(f"{name:>5s} Loss: {epoch_loss / (i + 1.0):.3f}")
                pbar.update(1)
        for i in range(n_examples):
            print(
                f"-\n input: {tokenizer_l1.decode(inputs[:, i])[0]}\n"
                f"target: {decoded_targets[i]}\n"
                f"output: {decoded_translations[i]}",
            )
        print("-")
    hypotheses = accelerator.gather_for_metrics(hypotheses)
    references = accelerator.gather_for_metrics(references)
    bleu = corpus_bleu(hypotheses, references).score
    return epoch_loss / len(loader), bleu
