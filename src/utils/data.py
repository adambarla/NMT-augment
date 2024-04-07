from functools import partial
import torch
from datasets import load_dataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from utils.random import PersistentRandom


def get_dataset(cfg):
    name = cfg.data.name
    lang = cfg.data.lang
    directory = cfg.data.dir
    dataset = load_dataset(name, lang, cache_dir=directory)
    return dataset


def get_subset(cfg, dataset):
    persistent_random = PersistentRandom(seed=cfg.seed)
    limited_size = min(len(dataset), cfg.subset_size or len(dataset))
    permutation = persistent_random.permutation(range(len(dataset))[:limited_size])
    return dataset.select(permutation)


def get_dataloaders(cfg, tokenizer, dataset):
    lang = cfg.data.lang
    l1 = lang[:2]
    l2 = lang[3:]
    max_length = cfg.max_length
    batch_size = cfg.batch_size
    col_fn_args = partial(
        collate_fn, tokenizer=tokenizer, max_length=max_length, l1=l1, l2=l2
    )

    train_dataloader = DataLoader(
        get_subset(cfg, dataset["train"]),
        batch_size=batch_size,
        collate_fn=col_fn_args,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        collate_fn=col_fn_args,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        collate_fn=col_fn_args,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers,
        shuffle=False,
    )
    dataloader_info = (
        f"DataLoaders are set up with the following configurations:\n"
        f"Train: samples={len(dataset['train']):<7d} batches={len(train_dataloader):<7d}\n"
        f"Valid: samples={len(dataset['validation']):<7d} batches={len(val_dataloader):<7d}\n"
        f"Test:  samples={len(dataset['test']):<7d} batches={len(test_dataloader):<7d}"
    )
    print(dataloader_info)
    return train_dataloader, val_dataloader, test_dataloader


def collate_fn(batch, tokenizer, max_length, l1, l2):
    src_batch, tgt_batch = [], []
    for item in batch:
        src_batch.append(
            tokenizer.encode(
                item["translation"][l1],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        )
        tgt_batch.append(
            tokenizer.encode(
                item["translation"][l2],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        )
    src_batch = pad_sequence(
        [torch.tensor(x) for x in src_batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    tgt_batch = pad_sequence(
        [torch.tensor(y) for y in tgt_batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    return src_batch, tgt_batch
