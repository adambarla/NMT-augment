import hydra
from utils import init_wandb, set_deterministic


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg):
    print(cfg)
    set_deterministic(cfg.seed)
    init_wandb(cfg)


if __name__ == "__main__":
    main()
