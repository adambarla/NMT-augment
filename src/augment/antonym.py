from augment.wordaugmenter import WordAugmenter
import nltk
from nltk.corpus import wordnet
import random, re
import math


class WordDictionary:
    def __init__(self, cache=True):
        self.cache = cache

    # pylint: disable=R0201
    def train(self, data):
        raise NotImplementedError

    # pylint: disable=R0201
    def predict(self, data):
        raise NotImplementedError

    # pylint: disable=R0201
    def save(self, model_path):
        raise NotImplementedError

    # pylint: disable=R0201
    def read(self, model_path):
        raise NotImplementedError


class PartOfSpeech:
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"

    pos2con = {
        "n": ["NN", "NNS", "NNP", "NNPS"],
        "v": ["VB", "VBD", "VBG", "VBN", "VBZ"],
        "a": ["JJ", "JJR", "JJS", "IN"],
        "s": ["JJ", "JJR", "JJS", "IN"],  # Adjective Satellite
        "r": ["RB", "RBR", "RBS"],  # Adverb
    }

    con2pos = {}
    poses = []
    for key, values in pos2con.items():
        poses.extend(values)
        for value in values:
            if value not in con2pos:
                con2pos[value] = []
            con2pos[value].append(key)

    @staticmethod
    def pos2constituent(pos):
        if pos in PartOfSpeech.pos2con:
            return PartOfSpeech.pos2con[pos]
        return []

    @staticmethod
    def constituent2pos(con):
        if con in PartOfSpeech.con2pos:
            return PartOfSpeech.con2pos[con]
        return []

    @staticmethod
    def get_pos():
        return PartOfSpeech.poses


class WordNet(WordDictionary):
    def __init__(self, lang, is_synonym=True):
        super().__init__(cache=True)

        self.lang = lang
        self.is_synonym = is_synonym

        try:
            import nltk
            from nltk.corpus import wordnet
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Missed nltk library. Install nltk by `pip install nltk`"
            )

        supported_langs = [
            "als", "arb", "bul", "cat", "cmn", "dan", "ell", "eng", "eus", "fin", "fra", "glg", "heb", "hrv", "ind", "isl", "ita", "ita_iwn", "jpn", "lit", "nld", "nno", "nob", "pol", "por", "ron", "slk", "slv", "spa", "swe", "tha", "zsm",]

        if lang not in supported_langs:
            raise ValueError(
                f"Language {lang} is not one of the supported wordnet languages."
            )

        # try:
        #     # Check whether wordnet package is downloaded
        #     wordnet.synsets('computer')
        #     # Check whether POS package is downloaded
        #     nltk.pos_tag('computer')
        # except LookupError:
        #     nltk.download('wordnet')
        #     nltk.download('averaged_perceptron_tagger')

        self.model = self.read()

    def read(self):
        try:
            wordnet.synsets("testing")
            return wordnet
        except LookupError:
            nltk.download("wordnet")
            nltk.download("omw-1.4")
            return wordnet

    def predict(self, word, pos=None):
        results = []
        for synonym in self.model.synsets(word, pos=pos, lang=self.lang):
            for lemma in synonym.lemmas(lang=self.lang):
                if self.is_synonym:
                    results.append(lemma.name())
                else:
                    for antonym in lemma.antonyms():
                        results.append(antonym.name())
        return results

    @classmethod
    def pos_tag(cls, tokens):
        try:
            results = nltk.pos_tag(tokens)
        except LookupError:
            nltk.download("averaged_perceptron_tagger")
            results = nltk.pos_tag(tokens)

        return results


class AntonymAug(WordAugmenter):
    # https://arxiv.org/pdf/1809.02079.pdf
    """
    Augmenter that leverage semantic meaning to substitute word.

    :param str lang: Language of your text. Default value is 'eng'.
            supported_langs = ['als', 'arb', 'bul', 'cat', 'cmn', 'dan', 'ell', 'eng', 'eus', 'fin', 'fra', 'glg', 'heb', 'hrv', 'ind', 'isl', 'ita', 'ita_iwn', 'jpn', 'lit', 'nld', 'nno', 'nob', 'pol', 'por', 'ron', 'slk', 'slv', 'spa', 'swe', 'tha', 'zsm']
    :param float aug_p: Percentage of word will be augmented.
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
        aug_p. Otherwise, using aug_max.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter
    """

    def __init__(
        self,
        name="antonym",
        aug_min=1,
        aug_max=10,
        aug_p=0.3,
        lang="eng",
        stopwords=None,
        tokenizer=None,
        reverse_tokenizer=None,
        stopwords_regex=None,
    ):
        super().__init__(
            aug_p=aug_p,
            aug_min=aug_min,
            aug_max=aug_max,
            stopwords=stopwords,
            tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer,
            device="cpu",
            stopwords_regex=stopwords_regex,
            include_detail=False,
        )
        self.lang = lang
        self.model = self.get_model(lang)

    def skip_aug(self, token_idxes, tokens):
        results = []
        for token_idx in token_idxes:

            if tokens[token_idx][1] not in ['VB', 'VBD', 'VBZ', 'VBG', 'VBN', 'VBP', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:
                continue

            if len(self.get_candidates(tokens, token_idx)) == 0:
                continue
            
            results.append(token_idx)
        return results
    

    def substitute(self, data):
        if not data or not data.strip():
            return data

        tokens = self.tokenizer(data) if self.tokenizer else data.split()
        original_tokens = tokens[:]

        pos = self.model.pos_tag(original_tokens)

        aug_candidates = self._get_aug_idxes(pos)
        if aug_candidates is None or len(aug_candidates) == 0:
            return data
        
        aug_idxes, candidates = zip(*aug_candidates)
        if aug_idxes is None or len(aug_idxes) == 0:
            return data

        for aug_idx, original_token in enumerate(original_tokens):
            # Skip if no augment for word
            if aug_idx not in aug_idxes:
                continue
            
            candidates = self.get_candidates(pos, aug_idx)

            if len(candidates) > 0:
                candidate = self.sample(candidates, 1)[0]
                candidate = candidate.replace("_", " ").replace("-", " ").lower()
                substitute_token = self.align_capitalization(original_token, candidate)

                tokens[aug_idx] = substitute_token

        return (
            self.reverse_tokenizer(tokens)
            if self.reverse_tokenizer
            else " ".join(tokens)
        )

    @classmethod
    def get_model(cls, lang):
        return WordNet(lang=lang, is_synonym=False)

    def generate_aug_cnt(self, size):
        if self.aug_max is not None:
            return min(self.aug_max, max(self.aug_min, math.ceil(size * self.aug_p)))
        else:
            return max(self.aug_min, min(size, math.ceil(size * self.aug_p)))

    def _get_aug_idxes(self, tokens):
        aug_cnt = self.generate_aug_cnt(len(tokens))
        word_idxes = self.pre_skip_aug(tokens)
        word_idxes = self.skip_aug(word_idxes, tokens)
        if len(word_idxes) == 0:
            return None

        aug_idexes = []
        for aug_idx in word_idxes:
            word_poses = PartOfSpeech.constituent2pos(tokens[aug_idx][1])
            candidates = []
            if word_poses is None or len(word_poses) == 0:
                candidates.extend(self.model.predict(tokens[aug_idx][0]))
            else:
                for word_pos in word_poses:
                    candidates.extend(self.model.predict(tokens[aug_idx][0], pos=word_pos))

            candidates = [c for c in candidates if c.lower() != tokens[aug_idx][0].lower()]

            if len(candidates) > 0:
                candidate = self.sample(candidates, 1)[0]
                aug_idexes.append((aug_idx, candidate))

        if len(aug_idexes) < aug_cnt:
            aug_cnt = len(aug_idexes)

        aug_idexes = self.sample(aug_idexes, aug_cnt)
        return aug_idexes
    
    def get_candidates(self, tokens, token_idx):
        original_token = tokens[token_idx][0]
        word_poses = PartOfSpeech.constituent2pos(tokens[token_idx][1])
        candidates = []
        if word_poses is None or len(word_poses) == 0:
            # Use every possible words as the mapping does not defined correctly
            candidates.extend(self.model.predict(tokens[token_idx][0]))
        else:
            for word_pos in word_poses:
                candidates.extend(self.model.predict(tokens[token_idx][0], pos=word_pos))

        candidates = [c for c in candidates if c.lower() != original_token.lower()]
        return candidates
    
    def pre_skip_aug(self, tokens):
        results = []
        for token_idx, token in enumerate(tokens):
            if self.stopwords and token in self.stopwords:
                continue
            if self.stopwords_regex and re.match(self.stopwords_regex, token):
                continue
            results.append(token_idx)
        return results

    def sample(self, population, k):
        return random.sample(population, k)

    def align_capitalization(self, original_token, substitute_token):
        if original_token.istitle():
            return substitute_token.capitalize()
        elif original_token.isupper():
            return substitute_token.upper()
        elif original_token.islower():
            return substitute_token.lower()
        else:
            return substitute_token


class ApplyAntonymAug:
    def __init__(
        self,
        l1="fr",
        l2="en",
        lang1="fra",
        lang2="eng",
        aug_max=10,
        name="Antonym_Aug",
        aug_min=1,
        aug_p=0.3,
        stopwords=None,
        tokenizer=None,
        reverse_tokenizer=None,
        stopwords_regex=None,
    ):
        self.aug_en = None
        self.aug_fr = None
        try:
            self.aug_en = AntonymAug(
                name=name,
                lang=lang1,
                aug_min=aug_min,
                aug_max=aug_max,
                aug_p=aug_p,
                stopwords=stopwords,
                tokenizer=tokenizer,
                reverse_tokenizer=reverse_tokenizer,
                stopwords_regex=stopwords_regex,
            )
        except ValueError:
            print(f"lang1 is set to {lang1}")
        try:
            self.aug_fr = AntonymAug(
                name=name,
                lang=lang2,
                aug_min=aug_min,
                aug_max=aug_max,
                aug_p=aug_p,
                stopwords=stopwords,
                tokenizer=tokenizer,
                reverse_tokenizer=reverse_tokenizer,
                stopwords_regex=stopwords_regex,
            )
        except ValueError:
            print(f"lang2 is set to {lang2}")
        self.l1 = l1
        self.l2 = l2

    def __call__(self, example):
        original_translation = example["translation"]
        if isinstance(original_translation, list):
            translations = []
            for translation in original_translation:
                if (
                    isinstance(translation, dict)
                    and self.l1 in translation
                    and self.l2 in translation
                ):
                    en_text = translation[self.l1]
                    fr_text = translation[self.l2]
                    augmented_en = (
                        self.aug_en.substitute(en_text) if self.aug_en else en_text
                    )
                    augmented_fr = (
                        self.aug_fr.substitute(fr_text) if self.aug_fr else fr_text
                    )
                    translations.append({self.l1: augmented_en, self.l2: augmented_fr})
                else:
                    translations.append(translation)
            return {"translation": translations}
        else:
            return example
