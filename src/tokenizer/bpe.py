from collections import Counter
import torch
from tqdm import tqdm
from utils import PersistentRandom


class BPETokenizer:
    def __init__(self, dataset, max_vocab_size=10000, fraction=0.01, seed=42, **kwargs):
        self.max_vocab_size = max_vocab_size
        self.seed = seed
        self.fraction = fraction
        self.special_tokens = ["<s>", "<pad>", "</s>", "<unk>"]
        self.bos_token_id = 255  # these values of utf-8 encoding are invalid
        self.pad_token_id = 254
        self.eos_token_id = 253
        self.unk_token_id = 252
        self.vocab, self.merges = self._create_vocab(dataset)
        self.vocab_size = len(self.vocab)

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
            pairs = self._get_pair_counts(encoded)
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
        special_token_ids = [
            self.bos_token_id,
            self.pad_token_id,
            self.eos_token_id,
            self.unk_token_id,
        ]
        if isinstance(x, torch.Tensor):
            x = x.tolist()
        if isinstance(x, list) and (not x or isinstance(x[0], int)):
            x = [x]
        decoded_text = [
            b"".join(
                self.vocab[t] for t in seq
                if t not in self.special_token_ids
            ).decode("utf-8", errors="replace")
            for seq in x
        ]
        return decoded_text

    def _consolidate_dataset(self, dataset):
        lst = list()
        pr = PersistentRandom(self.seed)
        space = " ".encode("utf-8")
        n = sum(len(partition) for partition in dataset.values())
        with tqdm(total=n, desc=f"unifying data") as pbar:
            for partition in dataset.values():
                for example in partition:
                    if pr.rand() < self.fraction:
                        for l_example in example["translation"].values():
                            s = l_example.encode("utf-8")
                            lst.extend(list(s) + list(space))
                        pbar.set_description(
                            f"unifying data, n_tokens={len(lst)}"
                        )
                    pbar.update(1)
        return lst

    def _create_tokens(self, lst):
        tok = set(range(256))
        new_tok = 256
        vocab = {key: bytes([key]) for key in tok}
        merges = dict()
        org_len = len(lst)
        tokens_to_generate = self.vocab_size - len(tok)
        with tqdm(total=tokens_to_generate, desc=f"creating BPE") as pbar:
            for i in range(tokens_to_generate):
                counts = self._get_pair_counts(lst)
                to_merge = counts.most_common(1)[0][0]
                lst = self._merge_tokens(lst, to_merge, new_tok)
                vocab[new_tok] = vocab[to_merge[0]] + vocab[to_merge[1]]
                merges[to_merge] = new_tok
                new_tok += 1
                pbar.update(1)
                pbar.set_description(
                    f"creating BPE, compression={org_len / len(lst):.1f}"
                )
        return vocab, merges

    def _create_vocab(self, dataset):
        # consolidate dataset to huge list of utf-8 bytes
        l, t = self._consolidate_dataset(dataset)
        # recursively merge most common token pairs until vocab is full
        vocab, merges = self._consolidate_tokens(l, t)
        vocab[self.unk_token_id] = b"<unk>"
        vocab[self.eos_token_id] = b"<s/>"
        vocab[self.pad_token_id] = b"<pad>"
        vocab[self.bos_token_id] = b"<s>"
        return vocab, merges

    @staticmethod
    def _get_pair_counts(tok_lst: list):
        pair_counts = Counter()
        pair_counts.update(zip(tok_lst, tok_lst[1:]))
        return pair_counts

    @staticmethod
    def _merge_tokens(lst: list, pair: tuple, new_tok: int):
        el1, el2 = pair
        i = 0
        new_lst = []
        while i < len(lst) - 1:
            if lst[i] == el1 and lst[i + 1] == el2:
                new_lst.append(new_tok)
                i += 2
            else:
                new_lst.append(lst[i])
                i += 1
        if i < len(lst):
            new_lst.append(lst[i])
        return new_lst
