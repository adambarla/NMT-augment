# Data Augmentation for Neural Machine Translation

In the global landscape of today, effective communication across languages is essential. Yet, low-resource languages encounter substantial barriers due to the scarcity of available data. These languages are at a disadvantage with limited access to automatic translation services, with some not being supported by leading online translation platforms such as Google Translate. The challenge extends to securing large, high-quality datasets, which are crucial for traditional deep-learning approaches that depend on extensive training data.

This project seeks to address a critical issue in the field of machine translation: the data scarcity for low-resource languages. We aim to evaluate and compare different data augmentation strategies, specifically focusing on improving machine translation for these underrepresented languages.

## Installation

Use python `3.10.13` in your preferred environment and run
```
pip install -r requirements.txt
```

## Training

We use [Accelerate](https://huggingface.co/docs/accelerate/en/index) to leverage hardware accelerators for mixed precision training, gradient accumulation. For logging we use [Weights & Biases](https://wandb.ai/site).

```bash
cd src
```
Create `accelerate` config:
```bash
accelerate config
```
Run training:
```bash
accelerate launch -m training
```

For changing hyperparameters we used [hydra](https://hydra.cc/docs/intro/). For example, to change the languages the model trains on, you can run:

```bash
accelerate launch training.py data.l1=de data.l2=en
```
Same goes for all other parameter defined in `src/conf/`.


## Hyperparameter Tuning

We use [wandb sweep](https://docs.wandb.ai/guides/integrations/hydra). Run
```bash
wandb sweep conf/sweep/<sweep_name>.yaml
```
which produces a command like this:
```
wandb agent ...
```
which will start the tuning. This way the tuning can be done on multiple machines at once.

## Data

We use WMT 2014, a collection of datasets used in shared tasks of the Ninth Workshop on Statistical Machine Translation. WMT 2014 English-to-German, English-to-French are some of the most common datasets from WMT
2014 for machine translation.

## Augmentation methods

We implement three text augmentation methods to expand our dataset in a low-resource setting. These methods are aimed at increasing the diversity of the data and improving model generalization. To configure which augmentation method to use, you can specify the augmenter by its corresponding name `synonym`, `backtrans`, `antonym`, or `null`, if no augmentation is needed.

1. **Synonym Replacement Augmentation**: This involves replacing words in the text with their synonyms while preserving the original meaning. This technique is inspired by the work on [Character-level Convolutional Networks for Text](https://arxiv.org/pdf/1509.01626.pdf). To specify the language for synonym replacement, include the `lang` argument followed by the language code (e.g., `eng` for English).

   - To augment French text with synonyms:
     ```
     augmenter=synonym augmenter.synonym.lang1=fra
     ```
   - To augment English text with synonyms:
     ```
     augmenter=synonym augmenter.synonym.lang2=eng
     ```

2. **Back Translation Augmentation**: Back translation involves translating the text from one language to another that may not necessarily correspond to the secondary language in the model, and then back to the original language. This can introduce variations in the text while retaining its semantic meaning. This method was first proposed in [Improving Neural Machine Translation Models with Monolingual Data](https://aclanthology.org/P16-1009/). To specify the language pair for back translation, use the `from_model` and `to_model` arguments followed by the corresponding model names. If one of the model names is set to `null`, only the other language will be augmented.

   - To augment French text using the specified translation models:
     ```
     augmenter=backtrans augmenter.backtrans.from_model1=Helsinki-NLP/opus-mt-fr-en augmenter.backtrans.to_model1=Helsinki-NLP/opus-mt-en-fr
     ```
   - To augment English text using the specified translation models:
     ```
     augmenter=backtrans augmenter.backtrans.from_model2=facebook/wmt19-en-de augmenter.backtrans.to_model2=facebook/wmt19-de-en
     ```

3. **Antonym Replacement Augmentation**: This method replaces words in the text with their antonyms, altering the meaning while preserving the structure of the sentence. To specify the language for antonym replacement, include the `lang` argument with the language code (e.g., `eng` for English). If one of the languages is set to `null`, only the other language will be augmented.

   - To augment French text with antonyms:
     ```
     augmenter=antonym augmenter.antonym.lang1=fra
     ```
   - To augment English text with antonyms:
     ```
     augmenter=antonym augmenter.antonym.lang2=eng
     ```
     
