from functools import partial
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def get_dataset(cfg):
    name = cfg.data.name
    lang = cfg.data.lang
    directory = cfg.data.dir
    dataset = load_dataset(name, lang, cache_dir=directory, trust_remote_code=True)
    return dataset


def get_dataloaders(cfg, tokenizer, dataset):
    lang = cfg.data.lang
    l1 = lang[:2]
    l2 = lang[3:]
    max_length = cfg.max_length
    batch_size = cfg.batch_size
    collate_fn_with_args = partial(
        collate_fn, tokenizer=tokenizer, max_length=max_length, l1=l1, l2=l2
    )
    train_dataloader = DataLoader(
        dataset["train"], batch_size=batch_size, collate_fn=collate_fn_with_args
    )
    val_dataloader = DataLoader(
        dataset["validation"], batch_size=batch_size, collate_fn=collate_fn_with_args
    )
    test_dataloader = DataLoader(
        dataset["test"], batch_size=batch_size, collate_fn=collate_fn_with_args
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
