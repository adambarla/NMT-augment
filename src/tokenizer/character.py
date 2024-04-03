import copy
import torch


class CharacterTokenizer:
    def __init__(self, dataset, **kwargs):
        self.special_tokens = ["<s>", "<pad>", "</s>", "<unk>"]
        self.vocab = self._create_vocab(dataset)
        self._stoi = {c: i for i, c in enumerate(self.vocab)}
        self._itos = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self._stoi["<pad>"]
        self.bos_token_id = self._stoi["<s>"]
        self.eos_token_id = self._stoi["</s>"]
        self.unk_token_id = self._stoi["<unk>"]

    def encode(
        self,
        x,
        add_special_tokens: bool = True,
        truncation: bool = True,
        padding="max_length",
        max_length: int = None,
    ):
        encoded = [self._stoi.get(c, self._stoi["<unk>"]) for c in x]
        if truncation and max_length is not None:
            encoded = encoded[: max_length - (2 if add_special_tokens else 0)]
        if add_special_tokens:
            encoded = [self._stoi["<s>"]] + encoded + [self._stoi["</s>"]]
        if padding == "max_length" and max_length is not None:
            encoded += [self._stoi["<pad>"]] * (max_length - len(encoded))
        return encoded

    def decode(self, x):
        special_token_ids = self.encode(self.special_tokens, add_special_tokens=False)
        if isinstance(x, torch.Tensor):
            x = x.tolist()
        if isinstance(x, list) and (not x or isinstance(x[0], int)):
            x = [x]
        decoded = [
            "".join([self._itos[c] for c in seq if c not in special_token_ids])
            for seq in x
        ]
        return decoded

    def _create_vocab(self, dataset):
        chars = set()
        for i in dataset:
            for r in dataset[i]:
                for t in r["translation"]:
                    chars.update(r["translation"][t])
        vocab = copy.deepcopy(self.special_tokens)
        vocab.extend(chars)
        return vocab
