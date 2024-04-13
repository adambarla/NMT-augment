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

WMT 2014 English-to-German. WMT 2014 is a collection of datasets used in shared tasks of the Ninth Workshop on Statistical Machine Translation. WMT 2014 English-to-German is one of the most common datasets from WMT
2014 for machine translation
