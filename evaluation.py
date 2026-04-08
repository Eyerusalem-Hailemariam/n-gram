"""Competition-style evaluation pipeline for N-gram language models."""

import argparse
import random
import nltk
nltk.data.path.append('.')

from data_preprocessing import get_tokenized_data, preprocess_data
from language_model import (
    NGramModelEvaluator,
    count_n_grams,
)
from language_model_smoothing import (
    calculate_perplexity_with_method,
    compare_smoothing_techniques,
)


def sentence_oov_ratio(sentence, unk_token="<unk>"):
    if not sentence:
        return 0.0
    unk_count = sum(1 for tok in sentence if tok == unk_token)
    return unk_count / len(sentence)


def bootstrap_ci_mean_diff(values_a, values_b, n_bootstrap=400, seed=87):
    """95% CI for mean(values_a - values_b) via bootstrap."""
    if not values_a or not values_b:
        return 0.0, 0.0, 0.0
    m = min(len(values_a), len(values_b))
    paired = [(values_a[i], values_b[i]) for i in range(m)]
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_bootstrap):
        sample = [paired[rng.randrange(m)] for _ in range(m)]
        mean_diff = sum(a - b for a, b in sample) / m
        diffs.append(mean_diff)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]
    point = sum(a - b for a, b in paired) / m
    return point, lo, hi


def method_sentence_perplexities(sentences, stack, vocab_size, method, **kwargs):
    """Compute one perplexity value per sentence for significance/slice analysis."""
    values = []
    for sent in sentences:
        pp, _ = calculate_perplexity_with_method([sent], stack, vocab_size, method=method, **kwargs)
        values.append(pp)
    return values


print("="*80)
print("N-GRAM LANGUAGE MODEL: COMPREHENSIVE TECHNICAL EVALUATION")
print("="*80)

parser = argparse.ArgumentParser(description="Evaluate N-gram language models.")
parser.add_argument(
    "--fast",
    action="store_true",
    help="Run advanced smoothing and tuning on smaller subsets for faster turnaround.",
)
parser.add_argument(
    "--advanced-sample-size",
    type=int,
    default=0,
    help="Optional cap for advanced smoothing/tuning sentence count (0 = auto/full).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=87,
    help="Random seed for reproducible splits, sampling, and bootstrap.",
)
args = parser.parse_args()

# ============ LOAD AND PREPROCESS DATA ============
print("\n[1/7] LOADING AND PREPROCESSING DATA")
print("-" * 80)

with open("./data/en_US.twitter.txt", "r", encoding="utf-8") as f:
    data = f.read()

tokenized_data = get_tokenized_data(data)
random.seed(args.seed)
random.shuffle(tokenized_data)

train_size = int(len(tokenized_data) * 0.8)
train_data = tokenized_data[:train_size]
test_data = tokenized_data[train_size:]

minimum_freq = 2
train_data_processed, test_data_processed, vocabulary = preprocess_data(
    train_data, test_data, minimum_freq
)
print("   → Boundary tokens in use: start=<s>, end=<e>")

print(f"Total sentences: {len(tokenized_data)}")
print(f"Training sentences: {len(train_data_processed):,}")
print(f"Test sentences: {len(test_data_processed)}")
print(f"Vocabulary size: {len(vocabulary)} unique words")
print(f"OOV handling: Using <unk> token for words with frequency < {minimum_freq}")

# Keep a validation split for hyperparameter tuning, then report final on held-out eval split.
val_size = max(1, int(len(test_data_processed) * 0.5))
val_data = test_data_processed[:val_size]
final_test_data = test_data_processed[val_size:]
if len(final_test_data) == 0:
    final_test_data = val_data
print(f"Validation sentences (for tuning): {len(val_data):,}")
print(f"Final held-out evaluation sentences: {len(final_test_data):,}")

if args.advanced_sample_size > 0:
    advanced_cap = args.advanced_sample_size
elif args.fast:
    advanced_cap = 1200
else:
    advanced_cap = 0

if advanced_cap > 0:
    tuning_data = val_data[: min(advanced_cap, len(val_data))]
    advanced_eval_data = final_test_data[: min(advanced_cap, len(final_test_data))]
    print(f"Fast advanced mode: ON (cap={advanced_cap} sentences)")
    print(f"Advanced smoothing eval sentences: {len(advanced_eval_data):,}")
    print(f"Hyperparameter tuning sentences: {len(tuning_data):,}")
else:
    tuning_data = val_data
    advanced_eval_data = final_test_data
    print("Fast advanced mode: OFF (using full validation/final splits)")

# ============ TRAIN N-GRAM MODELS ============
print("\n[2/7] TRAINING N-GRAM MODELS")
print("-" * 80)

n_gram_counts_list = []
for n in range(1, 4):
    n_gram_counts = count_n_grams(train_data_processed, n)
    n_gram_counts_list.append(n_gram_counts)
    print(f"{n}-gram counts: {len(n_gram_counts):,} unique n-grams")

vocabulary_size = len(vocabulary) + 2  # +2 for <e> and <unk>

# ============ DEMONSTRATE OOV HANDLING ============
print("\n[3/7] OUT-OF-VOCABULARY (OOV) HANDLING DEMONSTRATION")
print("-" * 80)

# Count OOV tokens in test set
oov_count = 0
total_tokens = 0
for sentence in test_data_processed:
    for token in sentence:
        total_tokens += 1
        if token == "<unk>":
            oov_count += 1

oov_percentage = (oov_count / total_tokens) * 100 if total_tokens > 0 else 0
print(f"Total tokens in test set: {total_tokens:,}")
print(f"OOV tokens (<unk>): {oov_count:,} ({oov_percentage:.2f}%)")
print(f"Vocabulary coverage: {100 - oov_percentage:.2f}%")
print(f"Fixed lexicon size: {len(vocabulary)} words + <unk> + <e> = {vocabulary_size} total")

# ============ BASELINE + ADD-K EVALUATION ============
print("\n[4/7] MLE BASELINES + ADD-K PERPLEXITY EVALUATION")
print("-" * 80)
print("\nLower perplexity is better.\n")

evaluator = NGramModelEvaluator(n_gram_counts_list, vocabulary, final_test_data)
results = evaluator.evaluate_all_models(k_values=[0.0, 0.5, 1.0, 2.0])
evaluator.print_evaluation_report()

print("\nSmoothing effectiveness vs MLE baseline:")
print(f"{'Model':<12} {'MLE k=0':<14} {'Best Add-k':<14} {'Improvement':<12}")
print("-" * 56)
for model_name in ["Unigram", "Bigram", "Trigram"]:
    baseline = results[model_name]["k=0.00"]["mean_perplexity"]
    best_addk = min(
        v["mean_perplexity"] for k, v in results[model_name].items() if k != "k=0.00"
    )
    improvement = ((baseline - best_addk) / baseline) * 100 if baseline > 0 else 0.0
    print(f"{model_name:<12} {baseline:<14.2f} {best_addk:<14.2f} {improvement:<11.2f}%")


print("\n[5/7] ADVANCED SMOOTHING COMPARISON (ALL MODELS)")
print("-" * 80)
zero_gram_counts = {(): sum(n_gram_counts_list[0].values())}
model_stacks = {
    "Unigram": [zero_gram_counts, n_gram_counts_list[0]],
    "Bigram": [zero_gram_counts, n_gram_counts_list[0], n_gram_counts_list[1]],
    "Trigram": [zero_gram_counts, n_gram_counts_list[0], n_gram_counts_list[1], n_gram_counts_list[2]],
}
advanced_results_by_model = {}
best_advanced_by_model = {}
for model_name, stack in model_stacks.items():
    print(f"\nAdvanced smoothing for {model_name}:")
    advanced_results = compare_smoothing_techniques(
        stack, vocabulary, advanced_eval_data, vocabulary_size, model_label=model_name
    )
    advanced_results_by_model[model_name] = advanced_results
    best_advanced_by_model[model_name] = min(
        advanced_results.items(), key=lambda kv: kv[1]["mean"]
    )

# ============ HYPERPARAMETER SEARCH ============
print("\n[6/7] HYPERPARAMETER SEARCH (VALIDATION -> FINAL TEST)")
print("-" * 80)
print("Searching interpolation lambdas and Kneser-Ney discounts on validation split for each model...")

if args.fast:
    lambda_grid = [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]
    discount_grid = [0.6, 0.75, 0.9]
else:
    lambda_grid = [
        [0.1, 0.9],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.4, 0.6],
    ]
    discount_grid = [0.4, 0.6, 0.75, 0.9]

tuning_summary = {}
for model_name, stack in model_stacks.items():
    best_interp = {"lambdas": None, "val_pp": float("inf")}
    for lambdas in lambda_grid:
        val_pp, _ = calculate_perplexity_with_method(
            tuning_data,
            stack,
            vocabulary_size,
            method="interpolation",
            lambdas=lambdas,
        )
        if val_pp < best_interp["val_pp"]:
            best_interp = {"lambdas": lambdas, "val_pp": val_pp}

    best_kn = {"discount": None, "val_pp": float("inf")}
    for discount in discount_grid:
        val_pp, _ = calculate_perplexity_with_method(
            tuning_data,
            stack,
            vocabulary_size,
            method="kneser_ney",
            discount=discount,
        )
        if val_pp < best_kn["val_pp"]:
            best_kn = {"discount": discount, "val_pp": val_pp}

    interp_test_pp, interp_test_std = calculate_perplexity_with_method(
        advanced_eval_data,
        stack,
        vocabulary_size,
        method="interpolation",
        lambdas=best_interp["lambdas"],
    )
    kn_test_pp, kn_test_std = calculate_perplexity_with_method(
        advanced_eval_data,
        stack,
        vocabulary_size,
        method="kneser_ney",
        discount=best_kn["discount"],
    )

    tuning_summary[model_name] = {
        "interp": {
            "best": best_interp,
            "test_pp": interp_test_pp,
            "test_std": interp_test_std,
        },
        "kn": {
            "best": best_kn,
            "test_pp": kn_test_pp,
            "test_std": kn_test_std,
        },
    }

for model_name in ["Unigram", "Bigram", "Trigram"]:
    print(f"\n{model_name} hyperparameter tuning:")
    interp_res = tuning_summary[model_name]["interp"]
    kn_res = tuning_summary[model_name]["kn"]
    print(f"  Best interpolation lambdas (validation): {interp_res['best']['lambdas']}")
    print(f"  Validation perplexity: {interp_res['best']['val_pp']:.2f}")
    print(f"  Final-test perplexity: {interp_res['test_pp']:.2f} (+/- {interp_res['test_std']:.2f})")
    print(f"  Best Kneser-Ney discount (validation): {kn_res['best']['discount']}")
    print(f"  Validation perplexity: {kn_res['best']['val_pp']:.2f}")
    print(f"  Final-test perplexity: {kn_res['test_pp']:.2f} (+/- {kn_res['test_std']:.2f})")

# ============ STATISTICAL + SLICE ANALYSIS ============
print("\n[7/7] STATISTICAL SIGNIFICANCE + OOV SLICE ANALYSIS")
print("-" * 80)

if args.fast:
    stat_cap = min(400, len(advanced_eval_data))
    bootstrap_iters = 200
else:
    stat_cap = min(1200, len(advanced_eval_data))
    bootstrap_iters = 400

stat_eval_data = advanced_eval_data[:stat_cap]
print(f"Significance sample size: {len(stat_eval_data):,} sentences")
print(f"Bootstrap iterations: {bootstrap_iters}")

print("\nBootstrap CI for perplexity improvement vs Add-k (lower is better):")
print(f"{'Model':<10} {'Best Adv':<22} {'Mean Δ(Add-k-Adv)':<20} {'95% CI':<24}")
print("-" * 80)

for model_name in ["Unigram", "Bigram", "Trigram"]:
    stack = model_stacks[model_name]
    best_technique = best_advanced_by_model[model_name][0]
    addk_values = method_sentence_perplexities(
        stat_eval_data, stack, vocabulary_size, method="add_k", k=1.0
    )
    if best_technique == "Linear Interpolation":
        best_l = tuning_summary[model_name]["interp"]["best"]["lambdas"]
        adv_values = method_sentence_perplexities(
            stat_eval_data, stack, vocabulary_size, method="interpolation", lambdas=best_l
        )
    elif best_technique == "Kneser-Ney":
        best_d = tuning_summary[model_name]["kn"]["best"]["discount"]
        adv_values = method_sentence_perplexities(
            stat_eval_data, stack, vocabulary_size, method="kneser_ney", discount=best_d
        )
    elif best_technique == "Stupid Backoff":
        adv_values = method_sentence_perplexities(
            stat_eval_data, stack, vocabulary_size, method="stupid_backoff", backoff_factor=0.4
        )
    else:
        adv_values = method_sentence_perplexities(
            stat_eval_data, stack, vocabulary_size, method="add_k", k=1.0
        )
    point, ci_lo, ci_hi = bootstrap_ci_mean_diff(
        addk_values, adv_values, n_bootstrap=bootstrap_iters, seed=args.seed
    )
    print(f"{model_name:<10} {best_technique:<22} {point:<20.2f} [{ci_lo:.2f}, {ci_hi:.2f}]")

low_oov = [s for s in stat_eval_data if sentence_oov_ratio(s) <= 0.05]
high_oov = [s for s in stat_eval_data if sentence_oov_ratio(s) >= 0.20]
print("\nOOV slice analysis (best advanced method per model):")
print(f"Low-OOV sentences (<=5% <unk>): {len(low_oov):,}")
print(f"High-OOV sentences (>=20% <unk>): {len(high_oov):,}")
print(f"{'Model':<10} {'Split':<10} {'Add-k':<12} {'Best Adv':<12}")
print("-" * 50)

for model_name in ["Unigram", "Bigram", "Trigram"]:
    stack = model_stacks[model_name]
    best_technique = best_advanced_by_model[model_name][0]
    for split_name, split_data in [("Low-OOV", low_oov), ("High-OOV", high_oov)]:
        if not split_data:
            print(f"{model_name:<10} {split_name:<10} {'n/a':<12} {'n/a':<12}")
            continue
        addk_pp, _ = calculate_perplexity_with_method(
            split_data, stack, vocabulary_size, method="add_k", k=1.0
        )
        if best_technique == "Linear Interpolation":
            best_l = tuning_summary[model_name]["interp"]["best"]["lambdas"]
            adv_pp, _ = calculate_perplexity_with_method(
                split_data, stack, vocabulary_size, method="interpolation", lambdas=best_l
            )
        elif best_technique == "Kneser-Ney":
            best_d = tuning_summary[model_name]["kn"]["best"]["discount"]
            adv_pp, _ = calculate_perplexity_with_method(
                split_data, stack, vocabulary_size, method="kneser_ney", discount=best_d
            )
        elif best_technique == "Stupid Backoff":
            adv_pp, _ = calculate_perplexity_with_method(
                split_data, stack, vocabulary_size, method="stupid_backoff", backoff_factor=0.4
            )
        else:
            adv_pp = addk_pp
        print(f"{model_name:<10} {split_name:<10} {addk_pp:<12.2f} {adv_pp:<12.2f}")

# ============ FINAL READINESS SUMMARY ============
print("\nFINAL READINESS SUMMARY")
print("-" * 80)
best_model, best_param, best_pp = evaluator.get_best_model()
print(f"Best Add-k setup: {best_model} with {best_param} -> perplexity {best_pp:.2f}")
for model_name in ["Unigram", "Bigram", "Trigram"]:
    best_technique = best_advanced_by_model[model_name]
    print(
        f"Best advanced smoothing ({model_name}): {best_technique[0]} -> perplexity "
        f"{best_technique[1]['mean']:.2f}"
    )
print("Ready-to-report checklist:")
print("✓ MLE in log-space for Unigram/Bigram/Trigram probability chains")
print("✓ Fixed lexicon and <unk> replacement policy")
print("✓ Advanced smoothing comparison (interpolation, backoff, Kneser-Ney)")
print("✓ Validation-driven hyperparameter search (lambdas, KN discount)")
print("✓ Bootstrap confidence intervals for improvement claims")
print("✓ OOV-slice robustness analysis")
print("✓ Held-out perplexity evaluation with baseline vs smoothed models")

# ============ SUMMARY AND RECOMMENDATIONS ============
print("\n" + "="*80)
