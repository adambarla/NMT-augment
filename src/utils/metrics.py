import wandb
from evaluate import load


def init_metrics(cfg):
    return {n: load(n) for n in cfg.metrics}


def calculate_metrics(results, metrics, hypotheses, references):
    list_of_references = [[r] for r in references]
    for n, m in metrics.items():
        result = m.compute(predictions=hypotheses, references=list_of_references)
        results[n] = result[n] if n == "meteor" else result["score"]


def log_metrics(results, name, step):
    for k, v in results.items():
        print(f"{name} {k}: {v:.4f}", end=" | ")
    print()
    for k, v in results.items():
        wandb.log({f"{name}_{k}": v}, step=step)
