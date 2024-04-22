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
    """
    Augmenter that back translates the input text to augment the text. For example, if the input text is in English, it can be translated to another language (not necessarily the other language used for MT) and then translated back to English to generate augmented text. We use the Helsinki-NLP models and facebook/wmt19-en-de and facebook/wmt19-de-en for back translation.

    :param str from_model_name: Any model from https://huggingface.co/models?filter=translation&search=Helsinki-NLP. As long as from_model_name matches with to_model_name. For example, if from_model_name is English to Japanese, then to_model_name should correspond to Japanese to English.
    :param str to_model_name: Any model from https://huggingface.co/models?filter=translation&search=Helsinki-NLP.
    :param str device: Default value is CPU. If value is CPU, it uses CPU for processing. If value is CUDA, it uses GPU for processing. Possible values include 'cuda' and 'cpu'. (May able to use other options)
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.Default value is False and suggesting to keep it as False if performance is the consideration.
    :param int batch_size: Batch size.
    :param int max_length: The max length of output text.
    :param str name: Name of this augmenter
    """

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
        self.aug_lang1 = None
        self.aug_lang2 = None
        try:
            self.aug_lang1 = BackTranslationAug(
                from_model_name=from_model1,
                to_model_name=to_model1,
                name="BackTranslationAug",
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                force_reload=force_reload,
            )
        except Exception as e:
            print("lang1 backtranslation is None")
        try:
            self.aug_lang2 = BackTranslationAug(
                from_model_name=from_model2,
                to_model_name=to_model2,
                name="BackTranslationAug",
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                force_reload=force_reload,
            )
        except Exception as e:
            print("lang2 backtranslation is None")
        self.l1 = l1
        self.l2 = l2

    def __call__(self, example):
        original_translation = example["translation"]

        if isinstance(original_translation, list):
            translations = []
            for translation in original_translation:

                if isinstance(translation, dict):
                    l1_text = translation[self.l1]
                    l2_text = translation[self.l2]

                    augmented_l1 = (
                        self.aug_lang1.substitute(l1_text)[0]
                        if self.aug_lang1
                        else l1_text
                    )
                    augmented_l2 = (
                        self.aug_lang2.substitute(l2_text)[0]
                        if self.aug_lang1
                        else l2_text
                    )
                    translations.append({self.l1: augmented_l1, self.l2: augmented_l2})
                else:
                    translations.append(translation)
            return {"translation": translations}
        else:
            return example
