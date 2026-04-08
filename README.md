# N-gram Language Model for Word Prediction

## Abstract

This project implements and evaluates an N-gram language model for next-word prediction on a large Twitter corpus. It supports unigram, bigram, and trigram models, fixed-lexicon OOV handling via `<unk>`, and multiple smoothing methods (Add-k, Linear Interpolation, Stupid Backoff, and Kneser-Ney). The code includes both intrinsic evaluation (perplexity on held-out test data) and an interactive CLI predictor.

## Introduction

Language modeling is a fundamental task in natural language processing with applications ranging from speech recognition to machine translation. N-gram models, despite their simplicity, remain competitive baselines and are widely used in various NLP tasks. This project focuses on implementing and analyzing N-gram models for the specific task of word prediction in social media context, using Twitter data as our corpus.

The primary objectives of this research are:
1. To implement an efficient N-gram model capable of handling large text corpora
2. To evaluate the performance of different N-gram orders (1 to 3) for word prediction
3. To compare multiple smoothing/backoff strategies on sparse social-media text
4. To provide an interactive interface for real-time word prediction

## Theoretical Background

### N-gram Language Models

An N-gram is a contiguous sequence of N items from a given text. In the context of language modeling, these items are typically words. The N-gram model approximates the probability of a word given its history by considering only the N-1 preceding words:

P(w_n | w_1^(n-1)) ≈ P(w_n | w_(n-N+1)^(n-1))

where w_i^j represents the sequence of words from position i to j.

### Smoothing

Smoothing techniques address zero probabilities for unseen N-grams. This project implements Add-k smoothing and advanced methods (Linear Interpolation, Stupid Backoff, Kneser-Ney). Add-k follows:

P(w_n | w_(n-N+1)^(n-1)) = (count(w_(n-N+1)^n) + k) / (count(w_(n-N+1)^(n-1)) + k|V|)

where |V| is the vocabulary size.

### Perplexity

Perplexity is used as an intrinsic evaluation metric for our language model. It is defined as:

PP(W) = P(w_1, w_2, ..., w_N)^(-1/N)

where W is a sequence of N words. Lower perplexity indicates better model performance.

## Methodology

Our approach consists of the following steps:

1. **Data Collection and Preprocessing**: We use a large corpus of English tweets. The data is tokenized and split into train/test, then test is split into validation and held-out final test.

2. **Vocabulary Building**: We construct a vocabulary from the training data, replacing infrequent words with an `<unk>` token to manage the vocabulary size.

3. **N-gram Extraction**: We extract N-grams of orders 1 to 3 from the processed training data.

4. **Probability Estimation**: We estimate N-gram probabilities with MLE (k=0) and smoothed variants for robust inference.

5. **Word Prediction**: Given a sequence of words, we predict the next word by calculating the probability of each word in the vocabulary and selecting the one with the highest probability.

6. **Model Evaluation**: We evaluate models via perplexity on held-out test data; validation is used for hyperparameter tuning (e.g., interpolation lambdas, KN discount).

## Implementation

The project is implemented in Python, leveraging libraries such as NLTK for tokenization and NumPy for efficient numerical computations. The main components of the implementation are:

1. **Data Preprocessing** (`data_preprocessing.py`):
    - Sentence splitting and tokenization
    - Vocabulary building with frequency thresholding
    - Replacement of out-of-vocabulary words with `<unk>` token

2. **N-gram Core** (`language_model.py`):
    - N-gram counting (1 to 3-gram)
    - MLE/Add-k probability estimation
    - Log-space perplexity utilities

3. **Advanced Smoothing** (`language_model_smoothing.py`):
    - Linear Interpolation
    - Stupid Backoff
    - Kneser-Ney

4. **Evaluation** (`evaluation.py`):
    - Intrinsic evaluation (perplexity leaderboard)
    - Validation-to-test hyperparameter search
    - OOV slice and significance-style comparisons

5. **Interactive Interface** (`main_interface.py`):
    - Data loading and model training
    - Interactive next-word prediction with debug scoring output

Key functions include:

- `count_n_grams()`: Extracts and counts N-grams from the corpus
- `estimate_probability()`: Calculates smoothed probability for a given word and context
- `suggest_with_kn_backoff()`: Scores and ranks candidate next words in interactive mode
- `calculate_perplexity()`: Computes the perplexity of the model on a given text

## Results and Evaluation

We evaluated our N-gram models (N=1 to 3) on a held-out test set. The results below come from the latest `python3 evaluation.py` run in this repository.

| Model | Perplexity |
|-------|------------|
| Unigram (MLE, k=0.00) | 590.67 |
| Bigram (MLE, k=0.00) | 74,545,300.26 |
| Trigram (MLE, k=0.00) | 86,258,909.35 |
| Trigram (Best Add-k, k=0.50) | 4,494.51 |
| Trigram (Best advanced: Kneser-Ney) | 360.04 |

Additional reference from the same run:
- Best Add-k overall: Unigram with k=1.00 → 588.01
- Best advanced Bigram: Kneser-Ney → 335.67
- Best advanced Trigram: Kneser-Ney → 360.04

The best model depends on corpus characteristics and smoothing strategy. Run `evaluation.py` to generate a reproducible leaderboard over MLE and smoothed settings.

Qualitative analysis shows that higher-order N-grams produce more contextually relevant suggestions, especially for domain-specific phrases common in social media text.

## Discussion

Our results demonstrate the trade-off between model complexity and performance in N-gram language models. While higher-order N-grams (especially 3-grams) show improved perplexity scores, they also require more computational resources and may suffer from data sparsity issues.

Add-k is a useful baseline, while advanced smoothing (especially Kneser-Ney and Stupid Backoff) often improves perplexity under sparse contexts.

The use of Twitter data introduces unique challenges, such as handling informal language, abbreviations, and hashtags. Future work could focus on developing preprocessing techniques specifically tailored to social media text.



## Installation and Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run comprehensive evaluation:

```bash
python3 evaluation.py
```

3. Run interactive predictor:

```bash
python3 main_interface.py
```
# n-gram
