from functools import partial
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from utils.random import PersistentRandom


def get_dataset(cfg):
    name = cfg.data.name
    lang = f"{cfg.data.l1}-{cfg.data.l2}"
    directory = cfg.data.dir
    dataset = load_dataset(name, lang, cache_dir=directory)
    return dataset


def get_subset(cfg, dataset):
    persistent_random = PersistentRandom(seed=cfg.seed)
    limited_size = min(len(dataset), cfg.subset_size or len(dataset))
    permutation = persistent_random.permutation(range(len(dataset)))[:limited_size]
    return dataset.select(permutation)


def get_augmented_subset(cfg, augmenter, dataset):

    subset = get_subset(cfg, dataset)
    if augmenter is not None:
        augmented_subset = subset.map(augmenter, batched=True, batch_size=1000)
        dataset = concatenate_datasets([subset, augmented_subset])  # TODO
    else:
        dataset = subset
    return dataset
def get_dataloaders(cfg, tokenizer_l1, tokenizer_l2, augmenter, dataset):
    col_fn_args = partial(
        collate_fn,
        tokenizer_l1=tokenizer_l1,
        tokenizer_l2=tokenizer_l2,
        max_length=cfg.max_length,
        l1=cfg.data.l1,
        l2=cfg.data.l2,
    )
    train_dataloader = DataLoader(
        get_augmented_subset(cfg, augmenter, dataset["train"]),
        batch_size=cfg.batch_size,
        collate_fn=col_fn_args,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset["validation"],
        batch_size=cfg.batch_size,
        collate_fn=col_fn_args,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        dataset["test"],
        batch_size=cfg.batch_size,
        collate_fn=col_fn_args,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers,
        shuffle=False,
    )
    dataloader_info = (
        f"Train Dataloader: samples={len(dataset['train']):<7d} batches={len(train_dataloader):<7d}\n"
        f"Valid Dataloader: samples={len(dataset['validation']):<7d} batches={len(val_dataloader):<7d}\n"
        f"Test  Dataloader: samples={len(dataset['test']):<7d} batches={len(test_dataloader):<7d}"
    )
    print(dataloader_info)
    return train_dataloader, val_dataloader, test_dataloader


def collate_fn(batch, tokenizer_l1, tokenizer_l2, max_length, l1, l2):
    src_batch, tgt_batch = [], []
    for item in batch:
        src_batch.append(
            tokenizer_l1.encode(
                item["translation"][l1],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        )
        tgt_batch.append(
            tokenizer_l2.encode(
                item["translation"][l2],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        )
    src_batch = pad_sequence(
        [torch.tensor(x) for x in src_batch],
        batch_first=True,
        padding_value=tokenizer_l1.pad_token_id,
    )
    tgt_batch = pad_sequence(
        [torch.tensor(y) for y in tgt_batch],
        batch_first=True,
        padding_value=tokenizer_l2.pad_token_id,
    )
    return src_batch, tgt_batch
