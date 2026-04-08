from collections import Counter, defaultdict
import math
import random
import numpy as np
import pandas as pd


# Function to generate n-grams
def count_n_grams(data, n, start_token="<s>", end_token="<e>"):
    """Count n-grams from tokenized sentences."""
    n_grams = Counter()
    for sentence in data:
        sent = [start_token] * max(0, n - 1) + sentence + [end_token]
        for i in range(len(sent) - n + 1):
            n_grams[tuple(sent[i : i + n])] += 1
    return dict(n_grams)


def estimate_probability(
    word,
    previous_n_gram,
    n_gram_counts,
    n_plus1_gram_counts,
    vocabulary_size,
    k=1.0,
):
    """P(word | previous_n_gram) with Add-k smoothing."""
    previous_n_gram = tuple(previous_n_gram)
    prev_count = n_gram_counts.get(previous_n_gram, 0)
    denom = prev_count + (k * vocabulary_size)
    if denom <= 0:
        return 1.0 / max(vocabulary_size, 1)
    joint = previous_n_gram + (word,)
    num = n_plus1_gram_counts.get(joint, 0) + k
    return num / denom


# Log-space probability calculation for numerical stability
def estimate_probability_log(
    word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0
):
    """Log-probability for numerical stability."""
    prob = estimate_probability(
        word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k
    )
    return math.log(max(prob, 1e-12))


# Function defined below loops over all words in vocabulary to calculate probabilities for all possible words.
def estimate_probabilities(
    previous_n_gram,
    n_gram_counts,
    n_plus1_gram_counts,
    vocabulary,
    end_token="<e>",
    unknown_token="<unk>",
    k=1.0,
):
    previous_n_gram = tuple(previous_n_gram)
    vocab = vocabulary + [end_token, unknown_token]
    vocab_size = len(vocab)
    probabilities = {}
    for word in vocab:
        probabilities[word] = estimate_probability(
            word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocab_size, k=k
        )
    return probabilities


# Function to get the count matrix
def make_count_matrix(n_plus1_gram_counts, vocabulary):
    # add <e> <unk> to the vocabulary
    # <s> is omitted since it should not appear as the next word
    vocabulary = vocabulary + ["<e>", "<unk>"]

    # obtain unique n-grams
    n_grams = []
    for n_plus1_gram in n_plus1_gram_counts.keys():
        n_gram = n_plus1_gram[0:-1]
        n_grams.append(n_gram)
    n_grams = list(set(n_grams))

    # mapping from n-gram to row
    row_index = {n_gram: i for i, n_gram in enumerate(n_grams)}
    # mapping from next word to column
    col_index = {word: j for j, word in enumerate(vocabulary)}

    nrow = len(n_grams)
    ncol = len(vocabulary)
    count_matrix = np.zeros((nrow, ncol))
    for n_plus1_gram, count in n_plus1_gram_counts.items():
        n_gram = n_plus1_gram[0:-1]
        word = n_plus1_gram[-1]
        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i, j] = count

    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocabulary)
    return count_matrix


# Function calculates the probabilities of each word given the previous n-gram, and stores this in matrix form.
def make_probability_matrix(n_plus1_gram_counts, vocabulary, k, unique_words=None):
    count_matrix = make_count_matrix(n_plus1_gram_counts, unique_words)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix


# Function calculates perplexity which is used as an evaluation metric for our model
def calculate_perplexity(
    sentence,
    n_gram_counts,
    n_plus1_gram_counts,
    vocabulary_size,
    start_token="<s>",
    end_token="<e>",
    k=1.0,
):
    """Perplexity for one sentence using Add-k in log-space for numerical stability."""
    n = len(next(iter(n_gram_counts.keys())))
    sent = [start_token] * n + sentence + [end_token]
    total_log_prob = 0.0
    token_count = 0
    for t in range(n, len(sent)):
        context = tuple(sent[t - n : t])
        word = sent[t]
        total_log_prob += estimate_probability_log(
            word, context, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=k
        )
        token_count += 1
    if token_count == 0:
        return float("inf")
    return math.exp(-total_log_prob / token_count)


def calculate_perplexity_batch(
    sentences,
    n_gram_counts,
    n_plus1_gram_counts,
    vocabulary_size,
    start_token="<s>",
    end_token="<e>",
    k=1.0,
):
    """Calculate average perplexity over a batch of sentences."""
    n = len(next(iter(n_gram_counts.keys())))

    def probability_fn(word, context):
        return estimate_probability(
            word, context, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=k
        )

    return calculate_perplexity_batch_from_probability_fn(
        sentences,
        context_size=n,
        probability_fn=probability_fn,
        start_token=start_token,
        end_token=end_token,
    )


def calculate_perplexity_batch_from_probability_fn(
    sentences,
    context_size,
    probability_fn,
    start_token="<s>",
    end_token="<e>",
):
    """Shared perplexity loop for any smoothing method via probability_fn(word, context)."""
    perplexities = []
    for sentence in sentences:
        sent = [start_token] * context_size + sentence + [end_token]
        total_log_prob = 0.0
        token_count = 0
        for t in range(context_size, len(sent)):
            context = tuple(sent[t - context_size : t])
            word = sent[t]
            prob = probability_fn(word, context)
            total_log_prob += math.log(max(prob, 1e-12))
            token_count += 1
        if token_count > 0:
            perplexities.append(math.exp(-total_log_prob / token_count))
    if not perplexities:
        return float("inf"), 0.0
    return float(np.mean(perplexities)), float(np.std(perplexities))


# Function to suggest a word
def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0, start_with=None):
    # length of previous words
    n = len(list(n_gram_counts.keys())[0])

    # From the words that the user already typed
    # get the most recent 'n' words as the previous n-gram
    previous_n_gram = previous_tokens[-n:]

    probabilities = estimate_probabilities(previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary, k=k)

    # Initialize suggested word to None
    # This will be set to the word with highest probability
    suggestion = None


    max_prob = 0

    # For each word and its probability in the probabilities dictionary:
    for word, prob in probabilities.items():  # complete this line

        # If the optional start_with string is set
        if start_with is not None:  # complete this line

            # Check if the beginning of word does not match with the letters in 'start_with'
            if not word.startswith(start_with):  # complete this line

                # if they don't match, skip this word (move onto the next word)
                continue  # complete this line

        # Check if this word's probability
        # is greater than the current maximum probability
        if prob > max_prob:  # complete this line

            # If so, save this word as the best suggestion (so far)
            suggestion = word

            # Save the new maximum probability
            max_prob = prob

    return suggestion, max_prob


def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts - 1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i + 1]

        suggestion = suggest_a_word(previous_tokens, n_gram_counts,
                                    n_plus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        suggestions.append(suggestion)
    return suggestions



class NGramModelEvaluator:
    """Comprehensive evaluation framework for N-gram language models."""
    
    def __init__(self, n_gram_counts_list, vocabulary, test_data, start_token='<s>', end_token='<e>'):
        self.n_gram_counts_list = n_gram_counts_list
        self.vocabulary = vocabulary
        self.test_data = test_data
        self.start_token = start_token
        self.end_token = end_token
        self.vocabulary_size = len(vocabulary) + 2  # +2 for <e> and <unk>
        self.results = {}
    
    def evaluate_all_models(self, k_values=[0.0, 0.5, 1.0, 2.0]):
        """Evaluate unigram, bigram, trigram next-word models with Add-k."""
        results = {}

        if not self.n_gram_counts_list:
            self.results = results
            return results

        unigram_counts = self.n_gram_counts_list[0]
        zero_gram_counts = {(): sum(unigram_counts.values())}

        model_specs = [("Unigram", zero_gram_counts, unigram_counts)]
        if len(self.n_gram_counts_list) >= 2:
            model_specs.append(("Bigram", self.n_gram_counts_list[0], self.n_gram_counts_list[1]))
        if len(self.n_gram_counts_list) >= 3:
            model_specs.append(("Trigram", self.n_gram_counts_list[1], self.n_gram_counts_list[2]))

        for model_name, n_gram_counts, n_plus1_gram_counts in model_specs:
            results[model_name] = {}

            for k in k_values:
                mean_pp, std_pp = calculate_perplexity_batch(
                    self.test_data,
                    n_gram_counts,
                    n_plus1_gram_counts,
                    self.vocabulary_size, self.start_token, self.end_token, k
                )
                results[model_name][f"k={k:.2f}"] = {
                    "mean_perplexity": mean_pp,
                    "std_perplexity": std_pp,
                }

        self.results = results
        return results
    
    def print_evaluation_report(self):
        """Print formatted evaluation report."""
        print("\n" + "="*70)
        print("N-GRAM LANGUAGE MODEL EVALUATION REPORT")
        print("="*70)
        
        for model_name, smoothing_results in self.results.items():
            print(f"\n{model_name.upper()}")
            print("-" * 70)
            print(f"{'Smoothing Parameter':<25} {'Mean Perplexity':<20} {'Std Dev':<15}")
            print("-" * 70)
            
            for param, metrics in smoothing_results.items():
                print(
                    f"{param:<25} {metrics['mean_perplexity']:<20.2f} {metrics['std_perplexity']:<15.2f}"
                )
        
        print("\n" + "="*70)
    
    def get_best_model(self):
        """Return the model configuration with lowest perplexity across Add-k and advanced methods."""
        best_pp = float('inf')
        best_config = None

        # Add-k results stored in self.results
        for model_name, smoothing_results in self.results.items():
            for param, metrics in smoothing_results.items():
                if param == "k=0.00":
                    continue  # skip unsmoothed baseline
                if metrics['mean_perplexity'] < best_pp:
                    best_pp = metrics['mean_perplexity']
                    best_config = (model_name, param, float(best_pp))

        # Also check advanced smoothing results if stored
        if hasattr(self, 'advanced_results'):
            for technique, metrics in self.advanced_results.items():
                if metrics['mean'] < best_pp:
                    best_pp = metrics['mean']
                    best_config = ("Trigram", technique, float(best_pp))

        return best_config

    def store_advanced_results(self, advanced_results):
        """Store results from compare_smoothing_techniques for best-model selection."""
        self.advanced_results = advanced_results
