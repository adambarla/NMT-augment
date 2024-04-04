from collections import Counter

class WordTokenizer:
    def __init__(self, dataset, max_vocab_size=10000, min_freq=1, **kwargs):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.vocab = self._create_vocab(dataset)
        self._stoi = {w: i for i, w in enumerate(self.vocab)}
        self._itos = {i: w for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self._stoi["<pad>"]
        self.unk_token_id = self._stoi["<unk>"]

    def encode(
        self,
        x,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=None,
    ):
        words = x.split()
        encoded = [self._stoi.get(w, self.unk_token_id) for w in words]
        if truncation and max_length is not None:
            encoded = encoded[: max_length - (2 if add_special_tokens else 0)]
        if add_special_tokens:
            encoded = [self._stoi["<s>"]] + encoded + [self._stoi["</s>"]]
        if padding == "max_length" and max_length is not None:
            encoded += [self.pad_token_id] * (max_length - len(encoded))
        return encoded

    def decode(self, x):
        decoded_text = " ".join([self._itos.get(w, "<unk>") for w in x])
        return decoded_text

    def _create_vocab(self, dataset):
        word_counts = Counter()
        for i in dataset:
            for r in dataset[i]:
                for t in r["translation"]:
                    word_counts.update(r["translation"][t].split())

        vocab = ["<s>", "<pad>", "</s>", "<unk>"]
        vocab.extend([word for word, count in word_counts.most_common(self.max_vocab_size - 4) if count >= self.min_freq])
        return vocab