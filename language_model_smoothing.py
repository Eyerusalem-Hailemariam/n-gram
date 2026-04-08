from collections import Counter, defaultdict
import math
import random
import numpy as np
import pandas as pd
from language_model import estimate_probability, calculate_perplexity_batch_from_probability_fn

_KN_CACHE = {}



def linear_interpolation_probability(
    word,
    previous_n_gram,
    n_gram_counts_list,
    n_plus1_gram_counts_list,
    vocabulary_size,
    lambdas=None,
):
    """Linear interpolation over unigram/bigram/trigram conditionals."""
    num_orders = len(n_gram_counts_list)
    if lambdas is None:
        # Slightly favor higher-order context by default
        lambdas = [0.2, 0.3, 0.5][:num_orders]
    else:
        lambdas = list(lambdas)
        if len(lambdas) == 2 and num_orders == 3:
            # Interpret as [bigram, trigram]; reserve leftover for unigram.
            rem = max(0.0, 1.0 - lambdas[0] - lambdas[1])
            lambdas = [rem, lambdas[0], lambdas[1]]
        elif len(lambdas) < num_orders:
            pad = [1e-6] * (num_orders - len(lambdas))
            lambdas = lambdas + pad
        elif len(lambdas) > num_orders:
            lambdas = lambdas[:num_orders]
    total = sum(lambdas)
    lambdas = [x / total for x in lambdas]

    history = tuple(previous_n_gram)
    prob = 0.0
    for i in range(num_orders):
        context_len = len(next(iter(n_gram_counts_list[i].keys())))
        if context_len == 0:
            context = ()
        else:
            context = history[-context_len:]
        p_i = estimate_probability(
            word,
            context,
            n_gram_counts_list[i],
            n_plus1_gram_counts_list[i],
            vocabulary_size,
            k=1.0,
        )
        prob += lambdas[i] * p_i
    return max(prob, 1e-12)


def stupid_backoff_probability(
    word,
    previous_n_gram,
    n_gram_counts_list,
    n_plus1_gram_counts_list,
    vocabulary_size,
    backoff_factor=0.4,
):
    """Stupid backoff probability score."""
    history = tuple(previous_n_gram)
    max_idx = len(n_gram_counts_list) - 1
    for i in range(max_idx, -1, -1):
        context_len = len(next(iter(n_gram_counts_list[i].keys())))
        context = history[-context_len:] if context_len > 0 else ()
        joint = context + (word,)
        joint_count = n_plus1_gram_counts_list[i].get(joint, 0)
        if joint_count > 0:
            denom = n_gram_counts_list[i].get(context, 0)
            if denom > 0:
                return (backoff_factor ** (max_idx - i)) * (joint_count / denom)
    return 1.0 / max(vocabulary_size, 1)


def kneser_ney_probability(
    word,
    previous_n_gram,
    n_gram_counts_list,
    n_plus1_gram_counts_list,
    vocabulary_size,
    discount=0.75,
    kn_cache=None,
):
    """Interpolated Kneser-Ney (practical approximation for trigram stack)."""
    history = tuple(previous_n_gram)

    # Build and reuse cached continuation/context-follower statistics.
    if kn_cache is None:
        cache_key = tuple(id(c) for c in n_plus1_gram_counts_list)
        if cache_key not in _KN_CACHE:
            context_unique_followers = []
            for counts in n_plus1_gram_counts_list:
                follower_counts = defaultdict(int)
                for gram in counts:
                    follower_counts[gram[:-1]] += 1
                context_unique_followers.append(follower_counts)

            bigram_counts = n_plus1_gram_counts_list[0]
            continuation_by_word = defaultdict(int)
            for bg in bigram_counts:
                continuation_by_word[bg[-1]] += 1

            _KN_CACHE[cache_key] = {
                "context_unique_followers": context_unique_followers,
                "continuation_by_word": continuation_by_word,
                "total_unique_bigrams": max(len(bigram_counts), 1),
            }
        kn_cache = _KN_CACHE[cache_key]

    p_cont = kn_cache["continuation_by_word"].get(word, 0) / kn_cache["total_unique_bigrams"]

    def kn_recursive(level, context):
        if level < 0:
            return p_cont
        n_counts = n_gram_counts_list[level]
        np1_counts = n_plus1_gram_counts_list[level]
        c_h = n_counts.get(context, 0)
        if c_h == 0:
            return kn_recursive(level - 1, context[1:] if context else ())
        c_hw = np1_counts.get(context + (word,), 0)
        unique_followers = kn_cache["context_unique_followers"][level].get(context, 0)
        lambda_h = (discount * unique_followers) / c_h
        lower_context = context[1:] if context else ()
        return max(c_hw - discount, 0.0) / c_h + lambda_h * kn_recursive(
            level - 1, lower_context
        )

    top_level = len(n_gram_counts_list) - 1
    top_context_len = len(next(iter(n_gram_counts_list[top_level].keys())))
    top_context = history[-top_context_len:] if top_context_len > 0 else ()
    return max(kn_recursive(top_level, top_context), 1e-12)


def calculate_perplexity_with_method(
    sentences,
    n_gram_counts_list,
    vocabulary_size,
    method="add_k",
    k=1.0,
    lambdas=None,
    backoff_factor=0.4,
    discount=0.75,
    start_token="<s>",
    end_token="<e>",
):
    """Perplexity for a trigram model stack with selectable smoothing."""
    if not n_gram_counts_list:
        return float("inf"), 0.0

    if len(n_gram_counts_list) == 1:
        unigram_counts = n_gram_counts_list[0]
        unigram_total = sum(unigram_counts.values())

        def unigram_probability_fn(word, _history):
            # For unigram-only stacks, advanced methods fall back to smoothed unigram scoring.
            k_uni = k if method == "add_k" else 1.0
            denom = unigram_total + (k_uni * vocabulary_size)
            if denom <= 0:
                return 1.0 / max(vocabulary_size, 1)
            return (unigram_counts.get((word,), 0) + k_uni) / denom

        return calculate_perplexity_batch_from_probability_fn(
            sentences,
            context_size=0,
            probability_fn=unigram_probability_fn,
            start_token=start_token,
            end_token=end_token,
        )

    n_gram_counts = n_gram_counts_list[:-1]  # contexts: (), (w1), (w1,w2)
    n_plus1_counts = n_gram_counts_list[1:]  # conditionals to predict next token
    context_lengths = [len(next(iter(c.keys()))) for c in n_gram_counts]
    kn_cache = None
    if method == "kneser_ney":
        # Prime KN cache once for this evaluation run.
        kneser_ney_probability(
            "<unk>",
            (),
            n_gram_counts,
            n_plus1_counts,
            vocabulary_size,
            discount=discount,
        )
        cache_key = tuple(id(c) for c in n_plus1_counts)
        kn_cache = _KN_CACHE.get(cache_key)

    trigram_context_len = context_lengths[-1]

    def probability_fn(word, history):
        if method == "add_k":
            return estimate_probability(
                word, history, n_gram_counts[-1], n_plus1_counts[-1], vocabulary_size, k=k
            )
        if method == "interpolation":
            return linear_interpolation_probability(
                word, history, n_gram_counts, n_plus1_counts, vocabulary_size, lambdas=lambdas
            )
        if method == "stupid_backoff":
            return stupid_backoff_probability(
                word,
                history,
                n_gram_counts,
                n_plus1_counts,
                vocabulary_size,
                backoff_factor=backoff_factor,
            )
        if method == "kneser_ney":
            return kneser_ney_probability(
                word,
                history,
                n_gram_counts,
                n_plus1_counts,
                vocabulary_size,
                discount=discount,
                kn_cache=kn_cache,
            )
        raise ValueError(f"Unknown method: {method}")

    return calculate_perplexity_batch_from_probability_fn(
        sentences,
        context_size=trigram_context_len,
        probability_fn=probability_fn,
        start_token=start_token,
        end_token=end_token,
    )


def compare_smoothing_techniques(
    n_gram_counts_list,
    vocabulary,
    test_data,
    vocabulary_size,
    model_label="Model",
    start_token="<s>",
    end_token="<e>",
):
    """Compare advanced smoothing methods on trigram evaluation."""
    results = {}

    print(f"Evaluating smoothing methods on {model_label}...")
    
    # Test multiple k values for Add-k smoothing
    k_values = [0.0, 0.5, 1.0, 2.0]
    best_addk_pp = float('inf')
    best_addk_k = None
    
    for k in k_values:
        addk_pp, addk_std = calculate_perplexity_with_method(
            test_data, n_gram_counts_list, vocabulary_size, method="add_k", k=k
        )
        results[f"Add-k (k={k:.1f})"] = {"mean": addk_pp, "std": addk_std}
        if addk_pp < best_addk_pp:
            best_addk_pp = addk_pp
            best_addk_k = k

    interp_pp, interp_std = calculate_perplexity_with_method(
        test_data,
        n_gram_counts_list,
        vocabulary_size,
        method="interpolation",
        lambdas=[0.2, 0.3, 0.5],
    )
    results["Linear Interpolation"] = {"mean": interp_pp, "std": interp_std}

    backoff_pp, backoff_std = calculate_perplexity_with_method(
        test_data,
        n_gram_counts_list,
        vocabulary_size,
        method="stupid_backoff",
        backoff_factor=0.4,
    )
    results["Stupid Backoff"] = {"mean": backoff_pp, "std": backoff_std}

    kn_pp, kn_std = calculate_perplexity_with_method(
        test_data,
        n_gram_counts_list,
        vocabulary_size,
        method="kneser_ney",
        discount=0.75,
    )
    results["Kneser-Ney"] = {"mean": kn_pp, "std": kn_std}

    print(f"\nSmoothing Technique Comparison ({model_label})")
    print("-" * 60)
    print(f"{'Technique':<30} {'Mean Perplexity':<20} {'Std Dev':<10}")
    print("-" * 60)

    for technique, metrics in sorted(results.items(), key=lambda kv: kv[1]["mean"]):
        print(f"{technique:<30} {metrics['mean']:<20.2f} {metrics['std']:<10.2f}")

    return results