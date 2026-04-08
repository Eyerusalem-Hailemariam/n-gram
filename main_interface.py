from main_functions import *
from language_model_smoothing import (
    calculate_perplexity_with_method,
    stupid_backoff_probability,
    linear_interpolation_probability,
)

def suggest_with_kn_backoff(previous_tokens, vocabulary, n_gram_counts_list, context_indexes,
                            vocab_set=None, unigram_top_words=None, unigram_total=None,
                            k=0.5, top_k=8, trigram_low_prob=5e-4, allowed_after=None,
                            debug=False, kn_discount=0.75,
                            scoring_method="Kneser-Ney", scoring_model="Trigram",
                            interpolation_lambdas=None):
    if vocab_set is None:
        vocab_set = set(vocabulary)
    vocab_size = len(vocabulary) + 2

    unigram_counts = n_gram_counts_list[0]
    bigram_ctx,   bigram_next   = n_gram_counts_list[0], n_gram_counts_list[1]
    trigram_ctx,  trigram_next  = n_gram_counts_list[1], n_gram_counts_list[2]
    zero_gram_counts = {(): sum(unigram_counts.values())}

    history2 = tuple(previous_tokens[-2:])
    history1 = tuple(previous_tokens[-1:])
    token_len = len(previous_tokens)

    candidates = set()
    if token_len >= 2:
        candidates.update(context_indexes[2].get(history2, set()))
        if len(candidates) < top_k:
            candidates.update(context_indexes[1].get(history1, set()))
    elif token_len == 1:
        candidates.update(context_indexes[1].get(history1, set()))

    if len(candidates) < top_k:
        if unigram_top_words is None:
            unigram_top_words = [g[-1] for g, _ in sorted(unigram_counts.items(), key=lambda kv: kv[1], reverse=True)[:200]]
        for word in unigram_top_words:
            candidates.add(word)
            if len(candidates) >= 600:
                break

    candidates = [w for w in candidates if w in vocab_set and is_valid_prediction_token(w)]

    prev_word = previous_tokens[-1] if previous_tokens else None
    candidates = filter_by_context(prev_word, candidates, vocab_set, allowed_after)

    if not candidates:
        return []

    if unigram_total is None:
        unigram_total = sum(unigram_counts.values())

    kn_ctx  = [bigram_ctx,  trigram_ctx]
    kn_next = [bigram_next, trigram_next]

    model_stack_map = {
        "Unigram": [zero_gram_counts, unigram_counts],
        "Bigram": [zero_gram_counts, unigram_counts, bigram_next],
        "Trigram": [zero_gram_counts, unigram_counts, bigram_next, trigram_next],
    }
    selected_stack = model_stack_map.get(scoring_model, model_stack_map["Trigram"])
    method_ctx = selected_stack[:-1]
    method_next = selected_stack[1:]
    resolved_interp_lambdas = interpolation_lambdas or [0.2, 0.3, 0.5]

    # Pre-compute diversity bonus denominator once (total unique bigram continuations)
    total_bigram_continuations = len(set(ng[-1] for ng in n_gram_counts_list[1]))

    scores = []
    for word in candidates:
        p_method = 0.0
        p_kn = kneser_ney_probability(word, history2, kn_ctx, kn_next, vocab_size, discount=kn_discount)
        p_3g = estimate_probability(word, history2, trigram_ctx,  trigram_next,  vocab_size, k=k)
        p_2g = estimate_probability(word, history1, bigram_ctx,   bigram_next,   vocab_size, k=k)
        p_1g = unigram_probability(word, unigram_counts, unigram_total, k, vocab_size)

        if scoring_method == "Kneser-Ney":
            p_method = kneser_ney_probability(
                word, tuple(previous_tokens), method_ctx, method_next, vocab_size, discount=kn_discount
            )
        elif scoring_method == "Stupid Backoff":
            p_method = stupid_backoff_probability(
                word, tuple(previous_tokens), method_ctx, method_next, vocab_size, backoff_factor=0.4
            )
        elif scoring_method == "Linear Interpolation":
            p_method = linear_interpolation_probability(
                word, tuple(previous_tokens), method_ctx, method_next, vocab_size, lambdas=resolved_interp_lambdas
            )
        elif scoring_method == "Add-k":
            if scoring_model == "Unigram":
                p_method = estimate_probability(word, (), zero_gram_counts, unigram_counts, vocab_size, k=k)
            elif scoring_model == "Bigram":
                p_method = estimate_probability(word, history1, bigram_ctx, bigram_next, vocab_size, k=k)
            else:
                p_method = estimate_probability(word, history2, trigram_ctx, trigram_next, vocab_size, k=k)

        # ── Step 1: pick best context-aware probability ───────────────────────
        # Use the highest-order signal available for the typed history length.
        if token_len >= 2:
            p_top = p_method
            if p_top < trigram_low_prob:
                p_ctx = (0.65 * p_3g + 0.25 * p_2g + 0.10 * p_1g) if p_3g >= trigram_low_prob \
                        else (0.70 * p_2g + 0.30 * p_1g) if p_2g >= trigram_low_prob else p_1g
            else:
                p_ctx = p_method
        elif token_len == 1:
            p_ctx = 0.85 * p_2g + 0.15 * p_1g
        else:
            p_ctx = p_1g

        # ── Step 2: log-space ranking formula ────────────────────────────────
        #
        #   score = log P(word | context)        ← contextual signal
        #         - α * log P(word)              ← penalise high-frequency words
        #         + β * diversity_bonus(word)    ← reward words seen in many contexts
        #
        # α = 0.5  → moderate penalty on "the", "and", "to"
        # β = 0.1  → mild reward for words with broad contextual coverage
        #
        # diversity_bonus = log(unique_bigram_contexts_for_word + 1)
        # This is the same intuition as Kneser-Ney continuation probability,
        # but applied at the ranking stage rather than the smoothing stage.

        ALPHA = 0.3   # unigram adjustment weight
        BETA  = 0.1   # diversity bonus weight
        MAX_UNIGRAM_BOOST = 1.2  # cap rare-word boost from -alpha*log(P_unigram)

        log_p_ctx = math.log(max(p_ctx, 1e-12))
        log_p_uni = math.log(max(p_1g,  1e-12))

        # Count how many distinct left-contexts this word appears in (bigram level)
        unique_contexts = sum(
            1 for ng in n_gram_counts_list[1] if ng[-1] == word
        )
        diversity_bonus = math.log(unique_contexts + 1)

        unigram_adjustment = -ALPHA * log_p_uni
        unigram_adjustment = min(unigram_adjustment, MAX_UNIGRAM_BOOST)

        score = log_p_ctx + unigram_adjustment + BETA * diversity_bonus

        # Floor: never let a score be -inf (happens when p_ctx rounds to 0)
        if not math.isfinite(score):
            continue

        scores.append((word, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    if debug:
        top_debug = scores[:top_k]
        W = 12
        print("\n" + "─" * 95)
        print(f"  {'WORD':<{W}} {'P(M)':<10} {'P(3g)':<10} "
              f"{'P(2g)':<10} {'P(1g)':<10} {'DIV':<6} {'SCORE':<10}")
        print("─" * 95)

        debug_rows = []
        for word, score in top_debug:
            _p_method = p_method = 0.0
            if scoring_method == "Kneser-Ney":
                _p_method = kneser_ney_probability(
                    word, tuple(previous_tokens), method_ctx, method_next, vocab_size, discount=kn_discount
                )
            elif scoring_method == "Stupid Backoff":
                _p_method = stupid_backoff_probability(
                    word, tuple(previous_tokens), method_ctx, method_next, vocab_size, backoff_factor=0.4
                )
            elif scoring_method == "Linear Interpolation":
                _p_method = linear_interpolation_probability(
                    word, tuple(previous_tokens), method_ctx, method_next, vocab_size, lambdas=resolved_interp_lambdas
                )
            elif scoring_method == "Add-k":
                if scoring_model == "Unigram":
                    _p_method = estimate_probability(word, (), zero_gram_counts, unigram_counts, vocab_size, k=k)
                elif scoring_model == "Bigram":
                    _p_method = estimate_probability(word, history1, bigram_ctx, bigram_next, vocab_size, k=k)
                else:
                    _p_method = estimate_probability(word, history2, trigram_ctx, trigram_next, vocab_size, k=k)
            _p_3g = estimate_probability(word, history2, trigram_ctx,  trigram_next,  vocab_size, k=k)
            _p_2g = estimate_probability(word, history1, bigram_ctx,   bigram_next,   vocab_size, k=k)
            _p_1g = unigram_probability(word, unigram_counts, unigram_total, k, vocab_size)
            _div  = sum(1 for ng in n_gram_counts_list[1] if ng[-1] == word)
            debug_rows.append((word, score, _p_method, _p_3g, _p_2g, _p_1g, _div))
            print(f"  {word:<{W}} {_p_method:<10.5f} {_p_3g:<10.5f} "
                  f"{_p_2g:<10.5f} {_p_1g:<10.5f} {_div:<6} {score:<10.4f}")

        print("─" * 95)
        print(f"  Context used: {list(previous_tokens)}")
        print(f"  History2={history2}  History1={history1}")
        print(
            f"  Candidates pool: {len(scores)} words scored  |  α={ALPHA}  β={BETA}  "
            f"max_unigram_boost={MAX_UNIGRAM_BOOST}"
        )
        print("─" * 95)

        # ── Step-by-step probability calculation for top 3 words ─────────────
        print("\n  HOW THE SCORE IS CALCULATED (top 3 words)")
        print("  Formula:  score = log P(ctx) + min(−α·log P(unigram), max_unigram_boost) + β·log(diversity+1)")
        print(f"            where α={ALPHA}, β={BETA}, max_unigram_boost={MAX_UNIGRAM_BOOST}\n")

        for word, score, p_m, p_3g, p_2g, p_1g, div in debug_rows[:3]:
            # Determine which context level drove p_ctx
            if token_len >= 2:
                p_top = p_m
                if p_top >= trigram_low_prob:
                    p_ctx_val = p_m
                    ctx_formula = f"P({scoring_method} | {scoring_model})"
                elif p_3g >= trigram_low_prob:
                    p_ctx_val = 0.65*p_3g + 0.25*p_2g + 0.10*p_1g
                    ctx_formula = f"0.65·P(3g) + 0.25·P(2g) + 0.10·P(1g)  [{scoring_method} too weak]"
                elif p_2g >= trigram_low_prob:
                    p_ctx_val = 0.70*p_2g + 0.30*p_1g
                    ctx_formula = f"0.70·P(2g) + 0.30·P(1g)  [backed off to bigram]"
                else:
                    p_ctx_val = p_1g
                    ctx_formula = f"P(1g)  [fully backed off to unigram]"
            elif token_len == 1:
                p_ctx_val = 0.85*p_2g + 0.15*p_1g
                ctx_formula = "0.85·P(2g) + 0.15·P(1g)"
            else:
                p_ctx_val = p_1g
                ctx_formula = "P(1g)"

            log_ctx  = math.log(max(p_ctx_val, 1e-12))
            log_uni  = math.log(max(p_1g,      1e-12))
            div_bon  = math.log(div + 1)

            print(f"  ┌─ '{word}'")
            print(f"  │  Step 1 — Context probability P(ctx)")
            print(f"  │           = {ctx_formula}")
            print(f"  │           = {p_ctx_val:.6f}")
            print(f"  │  Step 2 — log P(ctx)              = log({p_ctx_val:.6f}) = {log_ctx:.4f}")
            unigram_boost_raw = -ALPHA * log_uni
            unigram_boost_used = min(unigram_boost_raw, MAX_UNIGRAM_BOOST)
            print(f"  │  Step 3 — unigram boost (raw)     = −{ALPHA}·log({p_1g:.6f}) = {unigram_boost_raw:.4f}")
            print(f"  │  Step 4 — unigram boost (capped)  = min({unigram_boost_raw:.4f}, {MAX_UNIGRAM_BOOST}) = {unigram_boost_used:.4f}")
            print(f"  │  Step 5 — β·log(diversity+1)      = {BETA}·log({div}+1) = {BETA*div_bon:.4f}")
            print(f"  │  Step 6 — Final score             = {log_ctx:.4f} + {unigram_boost_used:.4f} + {BETA*div_bon:.4f}")
            print(f"  └─ score = {score:.4f}\n")

        print("─" * 95 + "\n")

    return dedup_rank_suggestions(scores, top_k=top_k)



print("Loading and preprocessing data...")
with open("./data/en_US.twitter.txt", "r", encoding="utf-8") as f:
    data = f.read()

tokenized_data = get_tokenized_data(data)
random.seed(87)
random.shuffle(tokenized_data)

train_size = int(len(tokenized_data) * 0.8)
train_data_processed, test_data_processed, vocabulary = preprocess_data(
    tokenized_data[:train_size], tokenized_data[train_size:], 2
)
print("   → Boundary tokens in use: start=<s>, end=<e>")

# Match evaluation.py protocol for advanced smoothing/tuning
val_size = max(1, int(len(test_data_processed) * 0.5))
val_data = test_data_processed[:val_size]
test_data = test_data_processed[val_size:]
if len(test_data) == 0:
    test_data = val_data

print(f"Training: {len(train_data_processed)} sentences | "
      f"Test: {len(test_data_processed)} sentences | "
      f"Vocab: {len(vocabulary)} words")
print(f"   → Validation data: {len(val_data)} sentences")
print(f"   → Test data: {len(test_data)} sentences")

print("\nTraining N-gram models (1-gram to 3-gram)...")
n_gram_counts_list = []
for n in range(1, 4):
    counts = count_n_grams(train_data_processed, n)
    n_gram_counts_list.append(counts)
    print(f"  {n}-gram: {len(counts):,} unique n-grams")

print("\nBoundary token proof (real runtime sample):")
if train_data_processed:
    sample_sentence = train_data_processed[0]
    for n in range(1, 4):
        expanded = ["<s>"] * max(0, n - 1) + sample_sentence + ["<e>"]
        print(f"  n={n} expanded sample: {expanded[:12]}{' ...' if len(expanded) > 12 else ''}")

for n, counts in enumerate(n_gram_counts_list, 1):
    boundary_items = [(gram, cnt) for gram, cnt in counts.items() if "<s>" in gram or "<e>" in gram]
    boundary_items = sorted(boundary_items, key=lambda x: x[0])[:3]
    if boundary_items:
        print(f"  {n}-gram boundary examples: {boundary_items}")

# Build data-driven context filter from training corpus
print("  Building data-driven context filter...")
allowed_after = build_context_filter(
    train_data_processed, n_gram_counts_list[1], min_count=3, top_n=200
)
print(f"  Context filter learned for {len(allowed_after):,} trigger words")

vocabulary_size = len(vocabulary) + 2

print("\n" + "="*70)
print("INTRINSIC EVALUATION: PERPLEXITY ANALYSIS")
print("="*70)
evaluator = NGramModelEvaluator(n_gram_counts_list, vocabulary, test_data)
results = evaluator.evaluate_all_models(k_values=[0.0, 0.5, 1.0, 2.0])
evaluator.print_evaluation_report()

best_model, best_param, best_pp = evaluator.get_best_model()
print(f"\nBest Model (Add-k only): {best_model} with {best_param} → Perplexity: {best_pp:.2f}")
print("(Advanced smoothing comparison below will update this)")

print("\n" + "="*70)
print("SMOOTHING TECHNIQUE COMPARISON")
print("="*70)
zero_gram_counts = {(): sum(n_gram_counts_list[0].values())}
model_stacks = {
    "Unigram": [zero_gram_counts, n_gram_counts_list[0]],
    "Bigram": [zero_gram_counts, n_gram_counts_list[0], n_gram_counts_list[1]],
    "Trigram": [zero_gram_counts, n_gram_counts_list[0], n_gram_counts_list[1], n_gram_counts_list[2]],
}

advanced_results_by_model = {}
for model_name, stack in model_stacks.items():
    print(f"\nAdvanced smoothing for {model_name}:")
    advanced_results_by_model[model_name] = compare_smoothing_techniques(
        stack, vocabulary, test_data, vocabulary_size, model_label=model_name
    )

best_adv_model = None
best_adv_technique = None
best_adv_pp = float("inf")
for model_name, model_results in advanced_results_by_model.items():
    for technique, metrics in model_results.items():
        if metrics["mean"] < best_adv_pp:
            best_adv_pp = metrics["mean"]
            best_adv_model = model_name
            best_adv_technique = technique

best_model, best_param, best_pp = evaluator.get_best_model()
if best_adv_pp < best_pp:
    best_model, best_param, best_pp = best_adv_model, best_adv_technique, best_adv_pp
print(f"\nBest Model (all methods): {best_model} — {best_param} → Perplexity: {best_pp:.2f}")

# Fixed interactive setup (reverted from global auto-switch)
interactive_model = "Trigram"
interactive_method = "Kneser-Ney"
interactive_kn_discount = 0.75
interactive_interp_lambdas = [0.2, 0.3, 0.5]

print("\nTuning interactive Kneser-Ney discount on validation slice...")
discount_grid = [0.4, 0.6, 0.75, 0.9]
tuning_data = val_data
best_kn_val_pp = float("inf")
for discount in discount_grid:
    val_pp, _ = calculate_perplexity_with_method(
        tuning_data,
        model_stacks[interactive_model],
        vocabulary_size,
        method="kneser_ney",
        discount=discount,
    )
    if val_pp < best_kn_val_pp:
        best_kn_val_pp = val_pp
        interactive_kn_discount = discount
print(
    f"Interactive KN discount selected: {interactive_kn_discount} "
    f"(validation perplexity: {best_kn_val_pp:.2f})"
)

print("Tuning interactive interpolation lambdas on validation slice...")
lambda_grid = [
    [0.1, 0.9],
    [0.2, 0.8],
    [0.3, 0.7],
    [0.4, 0.6],
]
best_interp_val_pp = float("inf")
for lambdas in lambda_grid:
    val_pp, _ = calculate_perplexity_with_method(
        tuning_data,
        model_stacks[interactive_model],
        vocabulary_size,
        method="interpolation",
        lambdas=lambdas,
    )
    if val_pp < best_interp_val_pp:
        best_interp_val_pp = val_pp
        interactive_interp_lambdas = lambdas
print(
    f"Interactive interpolation lambdas selected: {interactive_interp_lambdas} "
    f"(validation perplexity: {best_interp_val_pp:.2f})"
)

def get_user_input_suggestions(vocabulary, n_gram_counts_list, allowed_after,
                               k=0.5, top_k=8, kn_discount=0.75,
                               scoring_method="Kneser-Ney", scoring_model="Trigram",
                               interpolation_lambdas=None):
    vocab_set      = set(vocabulary)
    unigram_counts = n_gram_counts_list[0]
    unigram_total  = sum(unigram_counts.values())
    unigram_top    = [g[-1] for g, _ in sorted(unigram_counts.items(), key=lambda kv: kv[1], reverse=True)[:200]]
    context_indexes = {
        1: build_context_index(n_gram_counts_list[1]),
        2: build_context_index(n_gram_counts_list[2]),
    }
    vocab_size = len(vocabulary) + 2

    print("\n" + "="*70)
    print("INTERACTIVE WORD PREDICTION")
    print("  Commands:")
    print("  • Normal input      → next-word suggestions")
    print("  • debug:<sentence>  → show full scoring breakdown")
    print("  • q                 → quit")
    print("="*70 + "\n")

    while True:
        user_input = input("Enter a sentence: ").lower().strip()
        if user_input == "q":
            break

        is_debug = user_input.startswith("debug:")
        if is_debug:
            user_input = user_input[6:].strip()

        tokens = preprocess_user_input(
            user_input, vocab_set,
            n_gram_counts_list=n_gram_counts_list,
            smoothing_k=k, unigram_total=unigram_total,
        )

        # Feature 2: context-anomaly correction on known words
        tokens = context_anomaly_correction(
            tokens, vocab_set, n_gram_counts_list, k, unigram_total, vocab_size
        )

        # ── Fix 2: keep <unk> in context for scoring ─────────────────────────
        # "i dont" → ["i", "<unk>"] → model uses P(word | i, <unk>) not P(word | i)
        # For suggestion scoring we pass the full token list including <unk>
        context_tokens = [t if t in vocab_set else "<unk>" for t in tokens]

        suggestions = suggest_with_kn_backoff(
            context_tokens, vocabulary, n_gram_counts_list, context_indexes,
            vocab_set=vocab_set, unigram_top_words=unigram_top,
            unigram_total=unigram_total, k=k, top_k=top_k,
            allowed_after=allowed_after, debug=is_debug, kn_discount=kn_discount,
            scoring_method=scoring_method, scoring_model=scoring_model,
            interpolation_lambdas=interpolation_lambdas,
        )

        print("\nTop suggestions:")
        if suggestions:
            # Convert log-space scores to readable 0-1 probabilities via softmax
            raw_scores = [s for _, s in suggestions]
            max_s = max(raw_scores)
            exp_scores = [math.exp(s - max_s) for s in raw_scores]
            total = sum(exp_scores)
            for i, ((word, _), norm) in enumerate(zip(suggestions, exp_scores), 1):
                print(f"  {i}. {word} (probability: {norm/total:.4f})")
        else:
            print("  (no confident suggestion found)")
        print()


if __name__ == "__main__":
    trigram_best_k, trigram_param = best_k_for_model(results, "Trigram")
    runtime_k = trigram_best_k
    print(
        f"\nInteractive mode: {interactive_model} + {interactive_method} "
        f"(k={runtime_k:.2f}, KN discount={interactive_kn_discount})"
    )
    get_user_input_suggestions(vocabulary, n_gram_counts_list, allowed_after,
                               k=runtime_k, top_k=8, kn_discount=interactive_kn_discount,
                               scoring_method=interactive_method, scoring_model=interactive_model,
                               interpolation_lambdas=interactive_interp_lambdas)