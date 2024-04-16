import torch
import wandb
from tqdm import tqdm
from utils.metrics import calculate_metrics, log_metrics


def epoch_train(model, loader, optimizer, scheduler, criterion, accelerator, step):
    epoch_loss = 0.0
    model.train()
    with tqdm(total=len(loader), desc="train") as pbar:
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
            wandb.log({"lr": scheduler.get_last_lr()[0]}, step=step)
            epoch_loss += accelerator.gather(loss).mean().item()
            pbar.set_description(f"train loss: {(epoch_loss / (i + 1.0)):.3f}")
            pbar.update(1)
            step += 1
    if accelerator.is_main_process:
        results = {'loss': epoch_loss / len(loader)}
        log_metrics(results, 'train', step)


def epoch_evaluate(
    model,
    loader,
    criterion,
    accelerator,
    tokenizer_l1,
    tokenizer_l2,
    metrics,
    step,
    n_examples=3,
    name="valid",
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
                pbar.set_description(f"{name:>5s} loss: {epoch_loss / (i + 1.0):.3f}")
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
    results = {'loss': epoch_loss / len(loader)}
    if accelerator.is_main_process:
        calculate_metrics(results, metrics, hypotheses, references)
        log_metrics(results, name, step)
    return results
