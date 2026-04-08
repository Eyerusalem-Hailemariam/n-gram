import random
import nltk
import re
import math
import difflib
from collections import defaultdict, Counter

# Missing-apostrophe normalizations (common in social text).
CONTRACTION_FIXES = {
    "cant": "can't", "dont": "don't", "didnt": "didn't",
    "doesnt": "doesn't", "isnt": "isn't", "arent": "aren't",
    "wasnt": "wasn't", "werent": "weren't", "wont": "won't",
    "wouldnt": "wouldn't", "couldnt": "couldn't", "shouldnt": "shouldn't",
    "youre": "you're", "theyre": "they're", "weve": "we've",
    "ive": "i've", "im": "i'm", "ill": "i'll", "youll": "you'll",
    "thats": "that's", "theres": "there's", "heres": "here's", "whats": "what's",
}

# Tokens that are always invalid as predictions regardless of context.
_ALWAYS_BLOCK = {"n't", "''", "``", "#", "<unk>", "<e>", "<s>"}

nltk.data.path.append('.')

from data_preprocessing import get_tokenized_data, preprocess_data
from language_model import (
    count_n_grams, NGramModelEvaluator, estimate_probability,
)
from language_model_smoothing import (
    compare_smoothing_techniques, kneser_ney_probability,
)



def best_k_for_model(results, model_name):
    """Pick the best non-zero Add-k smoothing value for a given model."""
    candidates = [(k, v["mean_perplexity"]) for k, v in results[model_name].items()
                  if k != "k=0.00"]
    best_param, _ = min(candidates, key=lambda item: item[1])
    return float(best_param.split("=")[1]), best_param


def is_valid_prediction_token(token):
    """Keep only clean word-like tokens for user-facing output."""
    if token in {"<unk>", "<e>", "<s>", "n't", "''", "``", "#"}:
        return False
    if not re.search(r"[a-z0-9]", token):
        return False
    return bool(re.fullmatch(r"[a-z0-9]+(?:'[a-z0-9]+)?", token))


def dedup_rank_suggestions(suggestions, top_k=5):
    """Merge duplicate words, keep highest score, return top_k."""
    best = {}
    for word, score in suggestions:
        if not is_valid_prediction_token(word):
            continue
        best[word] = max(score, best.get(word, float('-inf')))
    return sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]


def unigram_probability(word, unigram_counts, unigram_total, k, vocabulary_size):
    """Smoothed unigram probability P(word)."""
    if unigram_total <= 0:
        return 1.0 / max(vocabulary_size, 1)
    return (unigram_counts.get((word,), 0) + k) / (unigram_total + k * vocabulary_size)


def build_context_index(n_plus1_gram_counts):
    """Map context tuple → set of observed next words for fast candidate lookup."""
    index = defaultdict(set)
    for gram in n_plus1_gram_counts:
        index[gram[:-1]].add(gram[-1])
    return index



def build_context_filter(train_data, bigram_counts, min_count=3, top_n=200):
    """
    Learn which words follow each trigger word directly from training data.

    For every word w in the corpus, collect the top-N most frequent
    successors with count >= min_count.  This replaces all hardcoded
    _AFTER_I / _AFTER_HAVE sets with corpus-derived knowledge.

    Returns:
        allowed_after: dict[str, set[str]]
            Maps trigger word → set of allowed successors.
            Only populated for words that appear as a left-context
            at least (min_count * 5) times total (i.e. common triggers).
    """
    # Count how often each (prev, next) pair occurs
    successor_counts = defaultdict(Counter)  # prev -> Counter(next -> count)
    for sentence in train_data:
        for prev, nxt in zip(sentence, sentence[1:]):
            successor_counts[prev][nxt] += 1

    allowed_after = {}
    for prev, counter in successor_counts.items():
        total = sum(counter.values())
        if total < min_count * 5:
            continue  # trigger word too rare to learn from
        # Keep successors that appear at least min_count times
        frequent = {w: c for w, c in counter.items() if c >= min_count}
        if not frequent:
            continue
        # Take top_n by frequency
        top = sorted(frequent, key=frequent.get, reverse=True)[:top_n]
        allowed_after[prev] = set(top)

    return allowed_after


def filter_by_context(prev_word, candidates, vocab_set, allowed_after=None):
    """
    Data-driven context filter.

    If we have learned successors for prev_word, restrict candidates to
    that set.  Falls back to the full candidate list if the filter would
    leave fewer than 3 options (prevents over-filtering on sparse data).

    Always strips tokens in _ALWAYS_BLOCK regardless of context.
    """
    # Always remove garbage tokens
    candidates = [w for w in candidates if w not in _ALWAYS_BLOCK]

    if allowed_after is None or prev_word not in allowed_after:
        return candidates  # no learned filter for this word → return as-is

    filtered = [w for w in candidates if w in allowed_after[prev_word]]
    return filtered if len(filtered) >= 3 else candidates

def context_anomaly_correction(tokens, vocab_set, n_gram_counts_list, k, unigram_total, vocab_size):
    """
    Detect tokens that are valid words but contextually wrong (e.g. "i cent").
    If P(token | prev) is below threshold, try spelling-similar alternatives
    and replace with the one that maximises P(candidate | prev).

    Returns a new token list with anomalies corrected.
    """
    if len(tokens) < 2 or n_gram_counts_list is None:
        return tokens

    bigram_ctx  = n_gram_counts_list[0]
    bigram_next = n_gram_counts_list[1]
    unigram_counts = n_gram_counts_list[0]
    ANOMALY_THRESHOLD = 1e-4   # below this → suspect the token
    MIN_WORD_COUNT_TO_SKIP = 8
    MIN_RATIO = 0.85
    MIN_IMPROVEMENT_FACTOR = 4.0

    corrected = list(tokens)
    for i in range(1, len(corrected)):
        word = corrected[i]
        prev = corrected[i - 1]
        if word not in vocab_set:
            continue  # already handled by OOV stripping

        # Don't auto-correct common/likely-valid words.
        if unigram_counts.get((word,), 0) >= MIN_WORD_COUNT_TO_SKIP:
            continue

        p = estimate_probability(word, (prev,), bigram_ctx, bigram_next, vocab_size, k=k)
        if p >= ANOMALY_THRESHOLD:
            continue  # looks fine in context

        # Low contextual probability → search for a better alternative
        candidates = difflib.get_close_matches(word, list(vocab_set), n=16, cutoff=0.60)
        if not candidates:
            continue

        best_word, best_score = word, math.log(max(p, 1e-12))
        for cand in candidates:
            if cand == word:
                continue
            p_cand = estimate_probability(cand, (prev,), bigram_ctx, bigram_next, vocab_size, k=k)
            ratio  = difflib.SequenceMatcher(None, word, cand).ratio()
            if ratio < MIN_RATIO:
                continue
            if p_cand < max(ANOMALY_THRESHOLD, p * MIN_IMPROVEMENT_FACTOR):
                continue
            score  = math.log(max(p_cand, 1e-12)) + 1.5 * ratio
            if score > best_score:
                best_score = score
                best_word  = cand

        if best_word != word:
            print(f"Context correction: '{word}' → '{best_word}' (low P({word}|{prev}))")
            corrected[i] = best_word

    return corrected



def _best_typo_match(normalized, prev_token, vocabulary_set, unigram_counts,
                     unigram_total, bigram_ctx, bigram_next, vocab_size, k):
    if not normalized or not normalized.replace("'", "").isalpha():
        return None, 0.0, 0.0
    vocab_list = list(vocabulary_set)
    raw_matches = difflib.get_close_matches(normalized, vocab_list, n=24, cutoff=0.52)
    if not raw_matches:
        return None, 0.0, 0.0
    scored = []
    for cand in raw_matches:
        ratio = difflib.SequenceMatcher(None, normalized, cand).ratio()
        if ratio < 0.56:
            continue
        if prev_token:
            p = estimate_probability(cand, (prev_token,), bigram_ctx, bigram_next, vocab_size, k=k)
        else:
            p = unigram_probability(cand, unigram_counts, unigram_total, k, vocab_size)
        score = math.log(max(p, 1e-12)) + 8.0 * ratio
        scored.append((score, ratio, cand))
    if not scored:
        return None, 0.0, 0.0
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_ratio, best_word = scored[0]
    if len(scored) == 1 and best_ratio < 0.68:
        return None, best_ratio, best_score
    second = scored[1][0] if len(scored) > 1 else best_score - 1.0
    if len(normalized) <= 3:
        if best_ratio < 0.72 or (best_score - second) < 0.35:
            return None, best_ratio, best_score
    if best_ratio < 0.62 or (best_score - second) < 0.25:
        return None, best_ratio, best_score
    return best_word, best_ratio, best_score


def preprocess_user_input(user_input, vocabulary_set, *, n_gram_counts_list=None,
                          smoothing_k=0.5, unigram_total=None, show_typo_notes=True):
    """
    Normalize user input for autocomplete:
    1. Expand contractions BEFORE tokenization (dont→don't, im→i'm)
       so NLTK sees the correct form and context is preserved.
    2. Typo correction via LM-guided spelling similarity.
    3. Unknown words kept as <unk> so context is NOT lost.
    """
    vocab_size = len(vocabulary_set) + 2
    if n_gram_counts_list is not None:
        unigram_counts = n_gram_counts_list[0]
        bigram_ctx, bigram_next = n_gram_counts_list[0], n_gram_counts_list[1]
        if unigram_total is None:
            unigram_total = sum(unigram_counts.values())
    else:
        bigram_ctx = bigram_next = unigram_counts = None

    # ── Fix 4: normalize contractions at string level before tokenization ────
    # This ensures "dont" → "don't" so NLTK tokenizes it correctly as one unit
    # and the model sees the right vocabulary token.
    normalized_input = user_input.lower().strip()
    for raw, fixed in CONTRACTION_FIXES.items():
        # Only replace whole words (word boundary match)
        normalized_input = re.sub(r'\b' + raw + r'\b', fixed, normalized_input)

    tokens = nltk.word_tokenize(normalized_input)
    cleaned = []
    for token in tokens:
        if not re.search(r"[a-z0-9']", token):
            continue
        if token in vocabulary_set:
            cleaned.append(token)
            continue
        normalized = re.sub(r"[^a-z0-9']", "", token)
        if normalized in vocabulary_set:
            cleaned.append(normalized)
            continue

        prev = cleaned[-1] if cleaned else None

        # Try LM-guided typo correction
        if n_gram_counts_list is not None and len(normalized) >= 3 and normalized.replace("'", "").isalpha():
            best, best_ratio, _ = _best_typo_match(
                normalized, prev, vocabulary_set, unigram_counts,
                unigram_total, bigram_ctx, bigram_next, vocab_size, smoothing_k,
            )
            if best is not None:
                if show_typo_notes:
                    print(f"Typo correction: '{normalized}' → '{best}'")
                cleaned.append(best)
                continue

        # Fallback spelling-only correction
        if len(normalized) >= 3 and normalized.replace("'", "").isalpha():
            match = difflib.get_close_matches(normalized, list(vocabulary_set), n=1, cutoff=0.82)
            if match:
                if show_typo_notes:
                    print(f"Typo correction: '{normalized}' → '{match[0]}' (spelling only)")
                cleaned.append(match[0])
                continue

        # ── Fix 2: keep unknown tokens as <unk> instead of dropping ──────────
        # This preserves context length so the model uses bigram/trigram history
        # rather than collapsing to a shorter (weaker) context.
        if len(normalized) >= 2:
            cleaned.append("<unk>")

    return cleaned