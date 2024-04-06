from collections import Counter
import torch
import copy


class WordTokenizer:
    def __init__(self, dataset, max_vocab_size=10000, min_freq=1, **kwargs):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.special_tokens = ["<s>", "<pad>", "</s>", "<unk>"]
        self.vocab = self._create_vocab(dataset)
        self._stoi = {w: i for i, w in enumerate(self.vocab)}
        self._itos = {i: w for i, w in enumerate(self.vocab)}
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
        if isinstance(x, str):
            words = x.split()
        elif isinstance(x, list):
            words = x
        else:
            raise TypeError("Input must be a string or a list.")

        encoded = [self._stoi.get(w, self.unk_token_id) for w in words]
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
            " ".join([self._itos[w] for w in seq if w not in special_token_ids])
            for seq in x
        ]
        return decoded_text

    def _create_vocab(self, dataset):
        word_counts = Counter()
        for i in dataset:
            for r in dataset[i]:
                for t in r["translation"]:
                    word_counts.update(r["translation"][t].split())

        vocab = copy.deepcopy(self.special_tokens)
        vocab.extend(
            [
                word
                for word, count in word_counts.most_common(
                    self.max_vocab_size - len(self.special_tokens)
                )
                if count >= self.min_freq
            ]
        )
        return vocab
