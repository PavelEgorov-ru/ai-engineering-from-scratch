"""Toy multi-token prediction (MTP) demo.

Trains a char-level model with parallel MTP heads (k=1, 2, 3) on a short
repeating corpus, then runs self-speculative decoding and measures
acceptance rate of drafted tokens at depths 2 and 3.

Stdlib only. The "transformer" is a trivial embedding + context-mix + linear
classifier -- enough to learn the corpus, not a real LLM. The bookkeeping
(MTP loss, draft generation, acceptance measurement) is faithful to
Gloeckle et al. (2024).

Run:  python main.py
"""

from __future__ import annotations

import math
import random


CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "a quick red fox runs past the sleepy dog. "
    "the brown dog chases the quick fox. "
    "the lazy fox jumps over the brown dog. "
) * 6

CONTEXT = 4
HIDDEN = 20
EPOCHS = 200
LR = 0.05
SEED = 7


def build_vocab(text: str) -> tuple[list[str], dict[str, int]]:
    chars = sorted(set(text))
    return chars, {c: i for i, c in enumerate(chars)}


def rand_matrix(rows: int, cols: int, scale: float, rng: random.Random) -> list[list[float]]:
    return [[rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [v / s for v in exps]


def argmax(xs: list[float]) -> int:
    best_i, best_v = 0, xs[0]
    for i, v in enumerate(xs):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


class MTPModel:
    """Shared trunk + K parallel heads.

    Trunk: embed(context tokens) -> mean-pool -> tanh(linear projection).
    Head i: HIDDEN -> VOCAB logits, predicting token at offset i.
    """

    def __init__(self, vocab_size: int, k: int, rng: random.Random):
        self.V, self.K, self.H, self.C = vocab_size, k, HIDDEN, CONTEXT
        self.embed = rand_matrix(vocab_size, HIDDEN, 0.1, rng)
        self.trunk = rand_matrix(HIDDEN, HIDDEN, 0.1, rng)
        self.heads = [rand_matrix(HIDDEN, vocab_size, 0.1, rng) for _ in range(k)]

    def hidden(self, ctx: list[int]) -> list[float]:
        pooled = [0.0] * self.H
        for t in ctx:
            for j, v in enumerate(self.embed[t]):
                pooled[j] += v
        n = max(1, len(ctx))
        pooled = [v / n for v in pooled]
        out = [0.0] * self.H
        for j in range(self.H):
            s = sum(pooled[i] * self.trunk[i][j] for i in range(self.H))
            out[j] = math.tanh(s)
        return out

    def head_logits(self, h: list[float], head_idx: int) -> list[float]:
        W = self.heads[head_idx]
        return [sum(h[i] * W[i][j] for i in range(self.H)) for j in range(self.V)]


def train(model: MTPModel, ids: list[int]) -> None:
    """Toy SGD: gradient descent on the head weights only.

    The trunk is frozen after init -- the goal is to show the MTP bookkeeping
    and measure head-quality as a function of depth, not to demonstrate
    state-of-the-art representation learning.
    """
    N = len(ids)
    for epoch in range(EPOCHS):
        total = 0.0
        steps = 0
        for t in range(model.C, N - model.K):
            ctx = ids[t - model.C:t]
            h = model.hidden(ctx)
            for i in range(1, model.K + 1):
                target = ids[t + i - 1]
                logits = model.head_logits(h, i - 1)
                probs = softmax(logits)
                total += -math.log(max(1e-12, probs[target]))
                for v in range(model.V):
                    grad = probs[v] - (1.0 if v == target else 0.0)
                    col = model.heads[i - 1]
                    for row in range(model.H):
                        col[row][v] -= LR * grad * h[row]
            steps += 1
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1:3d}  avg NLL (summed over k={model.K} heads) = {total / max(1, steps):.3f}")


def head_accuracy(model: MTPModel, ids: list[int]) -> list[float]:
    correct = [0] * model.K
    count = 0
    for t in range(model.C, len(ids) - model.K):
        ctx = ids[t - model.C:t]
        h = model.hidden(ctx)
        for i in range(1, model.K + 1):
            pred = argmax(model.head_logits(h, i - 1))
            if pred == ids[t + i - 1]:
                correct[i - 1] += 1
        count += 1
    return [c / max(1, count) for c in correct]


def self_speculative_decode(model: MTPModel, ids: list[int]) -> tuple[list[float], float]:
    """Run head-1 autoregressively and verify heads 2..K drafts.

    Returns acceptance rate per draft depth and effective tokens per step.
    """
    if model.K < 2:
        return [], 1.0
    accepted = [0] * (model.K - 1)
    attempts = [0] * (model.K - 1)
    total_tokens = 0
    steps = 0
    for t in range(model.C, len(ids) - model.K):
        ctx = ids[t - model.C:t]
        h = model.hidden(ctx)
        tok_1 = argmax(model.head_logits(h, 0))
        drafts = [argmax(model.head_logits(h, i)) for i in range(1, model.K)]
        committed = [tok_1]
        for i, d in enumerate(drafts):
            new_ctx = (ctx + committed)[-model.C:]
            verify_h = model.hidden(new_ctx)
            verifier_pick = argmax(model.head_logits(verify_h, 0))
            attempts[i] += 1
            if d == verifier_pick:
                accepted[i] += 1
                committed.append(d)
            else:
                break
        total_tokens += len(committed)
        steps += 1
    rates = [accepted[i] / max(1, attempts[i]) for i in range(model.K - 1)]
    eff_tokens = total_tokens / max(1, steps)
    return rates, eff_tokens


def run_k(chars: list[str], ids: list[int], k: int) -> None:
    print(f"\n--- MTP with k = {k} ---")
    model = MTPModel(len(chars), k, random.Random(SEED + k))
    train(model, ids)
    for i, a in enumerate(head_accuracy(model, ids), start=1):
        print(f"  head {i} top-1 accuracy (depth {i}) : {a:.3f}")
    rates, eff = self_speculative_decode(model, ids)
    for i, r in enumerate(rates, start=2):
        print(f"  acceptance at depth {i}           : {r:.3f}")
    print(f"  effective tokens / decode step   : {eff:.3f}  (ceiling = {float(k):.1f})")
    print(f"  theoretical speedup              : {eff:.2f}x over next-token decode")


def main() -> None:
    print("=" * 60)
    print("MULTI-TOKEN PREDICTION (TOY)")
    print("=" * 60)
    chars, stoi = build_vocab(CORPUS)
    ids = [stoi[c] for c in CORPUS]
    print(f"corpus length : {len(ids)} tokens")
    print(f"vocab size    : {len(chars)}")
    print(f"context window: {CONTEXT}")
    print(f"hidden dim    : {HIDDEN}")
    for k in (1, 2, 3):
        run_k(chars, ids, k)
    print()
    print("The k=1 row is the next-token baseline.")
    print("For k=2 and k=3 the acceptance rate at each deeper head")
    print("is the probability that the draft matches head-1's argmax.")
    print("Effective tokens/step is 1 + sum(acceptance rates), capped at k.")


if __name__ == "__main__":
    main()
