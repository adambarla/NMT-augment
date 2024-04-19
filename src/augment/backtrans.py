from augment.wordaugmenter import WordAugmenter
import augment.mtmodels as mt

BACK_TRANSLATION_MODELS = {}


def init_back_translation_model(
    from_model_name,
    to_model_name,
    device,
    force_reload=False,
    batch_size=32,
    max_length=None,
):
    global BACK_TRANSLATION_MODELS

    model_name = "_".join([from_model_name, to_model_name, str(device)])
    if model_name in BACK_TRANSLATION_MODELS and not force_reload:
        BACK_TRANSLATION_MODELS[model_name].batch_size = batch_size
        BACK_TRANSLATION_MODELS[model_name].max_length = max_length

        return BACK_TRANSLATION_MODELS[model_name]

    model = mt.MtTransformers(
        src_model_name=from_model_name,
        tgt_model_name=to_model_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )

    BACK_TRANSLATION_MODELS[model_name] = model
    return model


class BackTranslationAug(WordAugmenter):
    # https://arxiv.org/pdf/1511.06709.pdf

    def __init__(
        self,
        from_model_name="facebook/wmt19-en-de",
        to_model_name="facebook/wmt19-de-en",
        name="BackTranslationAug",
        device="cpu",
        batch_size=32,
        max_length=300,
        force_reload=False,
    ):
        super().__init__(
            name=name,
            aug_p=None,
            aug_min=None,
            aug_max=None,
            tokenizer=None,
            device=device,
            include_detail=False,
        )

        self.model = self.get_model(
            from_model_name=from_model_name,
            to_model_name=to_model_name,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        self.device = self.model.device

    def substitute(self, data, n=1):
        if not data:
            return data

        augmented_text = self.model.predict(data)
        return augmented_text

    @classmethod
    def get_model(
        cls,
        from_model_name,
        to_model_name,
        device="cuda",
        force_reload=False,
        batch_size=32,
        max_length=None,
    ):
        return init_back_translation_model(
            from_model_name, to_model_name, device, force_reload, batch_size, max_length
        )

    @classmethod
    def clear_cache(cls):
        global BACK_TRANSLATION_MODELS
        BACK_TRANSLATION_MODELS = {}


class ApplyBackTranslationAug:
    def __init__(
        self,
        from_model1="facebook/wmt19-en-de",
        to_model1="facebook/wmt19-de-en",
        from_model2="facebook/wmt19-en-de",
        to_model2="facebook/wmt19-de-en",
        l1="fr",
        l2="en",
        device="cpu",
        batch_size=32,
        max_length=300,
        force_reload=False,
    ):
        self.aug_lang1 = BackTranslationAug(
            from_model=from_model1,
            to_model=to_model1,
            name="BackTranslationAug",
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            force_reload=force_reload,
        )
        self.aug_lang2 = BackTranslationAug(
            from_model=from_model2,
            to_model=to_model2,
            name="BackTranslationAug",
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            force_reload=force_reload,
        )
        self.l1 = l1
        self.l2 = l2

    def __call__(self, example):
        original_translation = example["translation"]
        if isinstance(original_translation, list):
            translations = []
            for translation in original_translation:
                if isinstance(translation, dict):
                    en_text = translation[self.l1]
                    fr_text = translation[self.l2]
                    augmented_en = (
                        self.aug_lang1.substitute(en_text)[0]
                        if self.aug_lang1
                        else en_text
                    )
                    augmented_fr = (
                        self.aug_lang2.substitute(fr_text)[0]
                        if self.aug_lang1
                        else fr_text
                    )
                    translations.append({self.l1: augmented_en, self.l2: augmented_fr})
                else:
                    translations.append(translation)
            return {"translation": translations}
        else:
            return example
