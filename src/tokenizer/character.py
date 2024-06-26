import copy
import torch


class CharacterTokenizer:
    def __init__(self, dataset, lang, **kwargs):
        self.lang = lang
        self.special_tokens = ["<s>", "<pad>", "</s>", "<unk>"]
        self.vocab = self._create_vocab(dataset)
        self._stoi = {c: i for i, c in enumerate(self.vocab)}
        self._itos = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self._stoi["<pad>"]
        self.bos_token_id = self._stoi["<s>"]
        self.eos_token_id = self._stoi["</s>"]
        self.unk_token_id = self._stoi["<unk>"]
        self.special_token_ids = [
            self.bos_token_id,
            self.pad_token_id,
            self.eos_token_id,
            self.unk_token_id,
        ]

    def encode(
        self,
        x,
        add_special_tokens: bool = True,
        truncation: bool = True,
        padding="max_length",
        max_length: int = None,
    ):
        encoded = [self._stoi.get(c, self.unk_token_id) for c in x]
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
            "".join([self._itos[c] for c in seq if c not in self.special_token_ids])
            for seq in x
        ]

    def _create_vocab(self, dataset):
        chars = set()
        for i in dataset:
            for r in dataset[i]:
                chars.update(r["translation"][self.lang])
        vocab = copy.deepcopy(self.special_tokens)
        vocab.extend(chars)
        return vocab
