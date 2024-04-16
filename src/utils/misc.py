import hydra
import wandb
from accelerate import Accelerator
from omegaconf import OmegaConf
from datetime import datetime


def init_accelerator(cfg):
    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=1,
    )
    print(f"object cfg.wandb: {OmegaConf.to_object(cfg.wandb)}")
    return accelerator


def init_wandb(cfg, accelerator):
    if accelerator.is_main_process:
        if cfg.group is None:
            # group of the run determined by model
            g = cfg.model._target_.split(".")[-1]
            cfg.group = f"{g}"
        if cfg.name is None:
            t = datetime.now().strftime("%Y%m%d_%H%M%S")
            # add other parameters of name if needed
            tok = cfg.tokenizer._target_.split(".")[-1]
            l1 = cfg.data.l1
            l2 = cfg.data.l2
            cfg.name = f"{tok}_{l1}_{l2}_{t}"

        wandb.init(
            name=cfg.name,
            group=cfg.group,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )

def init_accelerate_tracker(cfg, accelerator):
    if cfg.group is None:
        # group of the run determined by model
        g = cfg.model._target_.split(".")[-1]
        cfg.group = f"{g}"
    if cfg.name is None:
        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        # add other parameters of name if needed
        tok = cfg.tokenizer._target_.split(".")[-1]
        l1 = cfg.data.l1
        l2 = cfg.data.l2
        cfg.name = f"{tok}_{l1}_{l2}_{t}"
    
    accelerator.init_trackers(
            project_name=cfg.wandb.project,
            config=cfg,
            init_kwargs={"wandb": {"entity": cfg.wandb.entity, "group": cfg.group, "name": cfg.name}}
    )

def init_augmenter(cfg):
    if cfg.augmenter is None:
        print("No augmentations used.")
        return None
    augmenter = hydra.utils.instantiate(cfg.augmenter)
    print(f"Augmentation: {augmenter}")
    return augmenter


def init_tokenizers(cfg, dataset):
    print(f"Tokenizer {cfg.data.l1}:")
    tok_l1 = hydra.utils.instantiate(cfg.tokenizer, dataset=dataset, lang=cfg.data.l1)
    print(f"Tokenizer {cfg.data.l2}:")
    tok_l2 = hydra.utils.instantiate(cfg.tokenizer, dataset=dataset, lang=cfg.data.l2)
    example = dataset["train"]["translation"][0]
    example = example[cfg.data.l1] + "\n" + example[cfg.data.l2]
    print(f"{tok_l1.lang} tok:\n{colorize_tokens(tok_l1.encode(example), tok_l1)}")
    print(f"{tok_l2.lang} tok:\n{colorize_tokens(tok_l2.encode(example), tok_l2)}")
    return tok_l1, tok_l2


def init_model(cfg, tokenizer_l1, tokenizer_l2, device):
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
    model.to(device)
    print(f"Model: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    return model


def colorize_tokens(token_list, tokenzier):
    """
    Colorize tokens for better visualization.
    :param token_list: list of tokens
    :param tokenzier:
    :return: string with colorized tokens
    """
    color_palette = [
        (243, 232, 228),
        (200, 226, 249),
        (228, 244, 180),
        (227, 224, 240),
        (248, 240, 164),
        (212, 243, 211),
        (243, 220, 180),
        (217, 246, 244),
        (212, 220, 220),
    ]
    string_list = tokenzier.decode([[t] for t in token_list])
    string_list = [s for s in string_list if s]
    s = ""
    for index, string in enumerate(string_list):
        r, g, b = color_palette[index % len(color_palette)]
        s += f"\x1b[48;2;{r};{g};{b}m{string}"
    return s + "\x1b[0m"
