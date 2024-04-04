class WordTokenizer:
    def __init__(self, dataset, **kwargs):
        self.vocab = self._create_vocab(dataset)
        self._stoi = {w: i for i, w in enumerate(self.vocab)}
        self._itos = {i: w for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self._stoi["<pad>"]
        print(f"my_Vocabulary size: {self.vocab_size}")
        print(f"my_Pad token ID: {self.pad_token_id}")

    def encode(
        self,
        x,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=None,
    ):

        #print(f"Input text: {x}")

        words = x.split()
        #print(f"Words: {words}")
        encoded = [self._stoi.get(w, self._stoi["<unk>"]) for w in words]
        #print(f"Encoded: {encoded}")
        if truncation and max_length is not None:
            encoded = encoded[: max_length - (2 if add_special_tokens else 0)]
        if add_special_tokens:
            encoded = [self._stoi["<s>"]] + encoded + [self._stoi["</s>"]]
        if padding == "max_length" and max_length is not None:
            encoded += [self._stoi["<pad>"]] * (max_length - len(encoded))
        return encoded

    def decode(self, x):
        #print(f"Word indices: {x}")
        decoded_text = " ".join([self._itos[w] for w in x])
        print(f"my_Decoded text: {decoded_text}")
        return decoded_text

    def _create_vocab(self, dataset):
        words = set()
        for i in dataset:
            for r in dataset[i]:
                for t in r["translation"]:
                    words.update(r["translation"][t].split())
        #print(f"Unique words: {words}")
        vocab = ["<s>", "<pad>", "</s>", "<unk>"]
        vocab.extend(words)
        #print(f"Vocabulary: {vocab}")
        return vocab