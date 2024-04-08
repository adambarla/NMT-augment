import hashlib
import os
import pickle
from collections import Counter
from time import time

import regex
import torch
from tqdm import tqdm
from utils import PersistentRandom


class BPETokenizer:
    def __init__(
        self,
        dataset,
        lang,
        max_vocab_size=10000,
        fraction=0.01,
        seed=42,
        save_path="../data/bpe/",
        override=False,
        **kwargs,
    ):
        self.split_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""" # gpt 4 tok.
        self.lang = lang
        self.vocab_size = max_vocab_size
        self.seed = seed
        self.fraction = fraction
        self.save_path = save_path
        self._init_special_tokens()
        self._init_tokenizer(dataset, override)

    def encode(
        self,
        x,
        add_special_tokens: bool = True,
        truncation: bool = True,
        padding="max_length",
        max_length: int = None,
    ):
        encoded = list(x.encode("utf-8"))
        while len(encoded) > 1:
            pairs = Counter(pair for pair in zip(x, x[1:]))
            to_merge = min(pairs, key=lambda k: self.merges.get(k, float("inf")))
            if to_merge not in self.merges:
                break
            replace = self.merges[to_merge]
            encoded = self._merge_tokens(encoded, to_merge, replace)
        if truncation and max_length is not None:
            encoded = encoded[: max_length - (2 if add_special_tokens else 0)]
        if add_special_tokens:
            encoded = [self.bos_token_id] + encoded + [self.eos_token_id]
        if padding == "max_length" and max_length is not None:
            encoded += [self.pad_token_id] * (max_length - len(encoded))
        return encoded

    def decode(self, x):
        if isinstance(x, torch.Tensor):
            x = x.tolist()
        if isinstance(x, list) and (not x or isinstance(x[0], int)):
            x = [x]
        return [
            b"".join(
                self.vocab[t] for t in seq if t not in self.special_token_ids
            ).decode("utf-8", errors="replace")
            for seq in x
        ]

    def _init_special_tokens(self):
        self.special_tokens = ["<s>", "<pad>", "</s>", "<unk>"]
        self.bos_token_id = 255  # these values of utf-8 encoding are invalid
        self.pad_token_id = 254
        self.eos_token_id = 253
        self.unk_token_id = 252
        self.special_token_ids = [
            self.bos_token_id,
            self.pad_token_id,
            self.eos_token_id,
            self.unk_token_id,
        ]

    def _init_tokenizer(self, dataset, override):
        self.filename = f"{self._get_hash(dataset)}.pkl"
        full_path = os.path.join(self.save_path, self.filename)
        if os.path.exists(full_path) and not override:
            print("Loading existing tokenizer from:", full_path)
            self._load(full_path)
        else:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            print("Creating new tokenizer configuration.")
            self._create(dataset)
            self._save(full_path)

    def _get_hash(self, dataset):
        hash_input = {
            "dataset_jhash": hashlib.sha256(
                pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL)
            ).hexdigest(),
            "vocab_size": self.vocab_size,
            "fraction": self.fraction,
            "seed": self.seed,
            "lang": self.lang,
            "split_pattern" : self.split_pattern
        }
        serialized_input = pickle.dumps(hash_input, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(serialized_input).hexdigest()

    def _save(self, full_path):
        data = {
            "split_pattern" : self.split_pattern,
            "lang": self.lang,
            "vocab": self.vocab,
            "merges": self.merges,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "special_token_ids": self.special_token_ids,
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
        }
        with open(full_path, "wb") as f:
            pickle.dump(data, f)
        print("Tokenizer configuration saved to:", full_path)

    def _load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.lang = data["lang"]
        self.split_pattern = data["split_pattern"]
        self.vocab = data["vocab"]
        self.merges = data["merges"]
        self.vocab_size = data["vocab_size"]
        self.special_tokens = data["special_tokens"]
        self.special_token_ids = data["special_token_ids"]
        self.bos_token_id = data["bos_token_id"]
        self.pad_token_id = data["pad_token_id"]
        self.eos_token_id = data["eos_token_id"]
        self.unk_token_id = data["unk_token_id"]

    def _create(self, dataset):
        # consolidate dataset to a list of lists of utf-8 bytes
        lst = self._consolidate_dataset(dataset)
        # recursively merge most common token pairs until vocab is full
        vocab, merges = self._create_tokens(lst)
        vocab[self.unk_token_id] = b"<unk>"
        vocab[self.eos_token_id] = b"<s/>"
        vocab[self.pad_token_id] = b"<pad>"
        vocab[self.bos_token_id] = b"<s>"
        self.vocab = vocab
        self.merges = merges
        if self.vocab_size != len(vocab):
            raise Exception(f"{self.vocab_size}, {len(vocab)}")

    def _consolidate_dataset(self, dataset):
        lst = list()
        pr = PersistentRandom(self.seed)
        n = sum(len(partition) for partition in dataset.values())
        with tqdm(total=n, desc=f"unifying data") as pbar:
            for partition in dataset.values():
                for example in partition:
                    if pr.rand() < self.fraction:
                        s = example['translation'][self.lang]
                        lst += [list(m.encode("utf-8")) for m in regex.findall(self.split_pattern, s)]
                        pbar.set_description(f"unifying data, fraction={self.fraction}, n_tokens={len(lst)}")
                    pbar.update(1)
        return lst

    def _create_tokens(self, lst):
        tok = set(range(256))
        new_tok = 256
        vocab = {key: bytes([key]) for key in tok}
        merges = dict()
        tokens_to_generate = self.vocab_size - len(tok)
        with tqdm(total=tokens_to_generate, desc=f"creating BPE") as pbar:
            for i in range(tokens_to_generate):
                counts = Counter(pair for l in lst for pair in zip(l, l[1:]))
                to_merge = counts.most_common(1)[0][0]
                for i in range(len(lst)):
                    lst[i] = self._merge_tokens(lst[i], to_merge, new_tok)
                vocab[new_tok] = vocab[to_merge[0]] + vocab[to_merge[1]]
                merges[to_merge] = new_tok
                new_tok += 1
                pbar.update(1)
                pbar.set_description(
                    f"creating BPE, merged "
                    f"({vocab[to_merge[0]].decode('utf-8',errors='replace')},"
                    f"{vocab[to_merge[1]].decode('utf-8', errors='replace')})"
                )
        return vocab, merges

    @staticmethod
    def _merge_tokens(l: list, pair: tuple, new_tok: int):
        el1, el2 = pair
        new_l = []
        i = 0
        while i < len(l) - 1:
            if l[i] == el1 and l[i+1] == el2:
                new_l.append(new_tok)
                i += 2
            else:
                new_l.append(l[i])
                i += 1
        if i < len(l):
            new_l.append(l[-1])
        return new_l
