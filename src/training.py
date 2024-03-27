import hydra
from transformers import BertTokenizer
from utils import init_wandb, set_deterministic, get_dataloaders
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg):
    print(f"Hydra configuration:\n{OmegaConf.to_yaml(cfg)}")
    set_deterministic(cfg.seed)
    init_wandb(cfg)

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-multilingual-cased"
    )  # todo: change to hydrainstantiable
    train_loader, val_loader, test_loader = get_dataloaders(cfg, tokenizer)


if __name__ == "__main__":
    main()
