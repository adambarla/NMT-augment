class CharacterTokenizer:
    def __init__(self, dataset, **kwargs):
        self.vocab = self._create_vocab(dataset)
        self._stoi = {c: i for i, c in enumerate(self.vocab)}
        self._itos = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self._stoi["<pad>"]

    def encode(
        self,
        x,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=None,
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
        return "".join([self._itos[c] for c in x])

    def _create_vocab(self, dataset):
        chars = set()
        for i in dataset:
            for r in dataset[i]:
                for t in r["translation"]:
                    chars.update(r["translation"][t])
        vocab = ["<s>", "<pad>", "</s>", "<unk>"]
        vocab.extend(chars)
        return vocab
