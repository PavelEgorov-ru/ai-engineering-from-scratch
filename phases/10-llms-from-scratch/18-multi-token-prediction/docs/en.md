# Multi-Token Prediction (MTP)

> Next-token prediction trains on one label per position. Multi-token prediction trains on several future tokens at the same position, using independent or sequential heads on top of a shared trunk. You get denser training signal, mild downstream gains at scale, and a pre-built draft model you can reuse for self-speculative decoding at inference time.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10, Lessons 04, 12 (Pre-training mini-GPT, Inference optimization)
**Time:** ~45 minutes

## Learning Objectives

- Describe the MTP training objective and contrast it with vanilla next-token prediction
- Distinguish parallel MTP heads (Gloeckle et al., 2024) from DeepSeek-V3's sequential MTP modules
- Explain why predicted tokens at depths k=2, 3, ... are used as speculative drafts at inference time
- Measure acceptance rate of drafted tokens and reason about its effect on decoding speed

## The Problem

In Lesson 04 you trained a transformer with next-token cross-entropy. At position t the loss pulls the head toward token t+1. That is one scalar gradient signal per position. The rest of the context -- tokens t+2, t+3, which the model will have to generate next -- contribute nothing to the loss at t. You train on sequences of length 1024 and use roughly 1024 gradient signals per sequence. Half of the information in the batch is thrown away.

At inference the same model decodes one token per forward pass. A 70B model that pushes 50 tokens per second is sitting on 95% of its compute budget idle -- decode is memory-bandwidth bound, not compute bound. The arithmetic units of an H100 process an entire batch of logits in the same wall-clock time as a single token. Something is wrong.

Multi-token prediction attacks both problems with the same change. At each position you ask the model to predict the next k tokens, not just the next one, using k output heads on top of a shared trunk. Training signal becomes k times denser. At inference you skip the training loss and instead use the extra heads as a drop-in draft model for speculative decoding -- no second network, no distillation stage, no architecture surgery.

## The Concept

### The standard objective, rewritten

For a sequence `x_1, ..., x_T` with trunk embedding `h_t` at position t, next-token prediction (NTP) minimizes:

```
L_NTP = - sum_t log P(x_{t+1} | h_t)
```

One head, one future token, one loss term per position.

Multi-token prediction (MTP) replaces this with a sum over k future offsets:

```
L_MTP = - sum_t sum_{i=1..k} log P(x_{t+i} | h_t)
```

k heads, k future tokens, k loss terms per position. Same trunk, same compute until the final projection. The Gloeckle et al. (2024) paper calls the heads "independent" because each head predicts its depth directly from the shared hidden state `h_t`.

### Parallel heads (Meta, 2024)

Gloeckle et al. add k linear heads on the shared trunk. Each head i predicts `x_{t+i}` given `h_t`. During training all k heads fire at every position; during inference only head 1 is used for the autoregressive loop, and heads 2..k are optional drafts for speculative decoding.

The trunk does extra work implicitly. To make head 3 accurate, the representation at position t has to encode enough about the next three tokens simultaneously. The model stops being myopic -- it learns to "plan" two or three tokens ahead because the loss forces it to.

Empirically this buys:

- No overhead in training time (the heads are tiny linear layers).
- 12-17% absolute gain on HumanEval and MBPP for code models at 7B and above.
- No improvement on small models (< ~1B parameters) -- capacity below threshold, the extra heads just smear gradients.

### Sequential modules (DeepSeek-V3, 2024)

DeepSeek-V3 kept the MTP objective but changed the head topology. Instead of k independent heads reading the same `h_t`, it chains k small transformer modules. Module i reads the output of module i-1 plus the embedding of the token it predicted, so the causal chain between predicted tokens is preserved.

```
depth 1: h_t         -> x_{t+1}
depth 2: [h_t; emb(x_{t+1})] -> mini-transformer -> x_{t+2}
depth 3: [prev;      emb(x_{t+2})] -> mini-transformer -> x_{t+3}
```

This is more parameters per MTP head (DeepSeek-V3's 14B of MTP weights sit on top of the 671B main model) and more compute per training step, but the heads' drafts are higher quality because each one conditions on the actual previous draft rather than predicting its depth cold. The distinction is the same one Hydra (Ankner et al., 2024) drew against Medusa: sequentially-dependent drafts beat sequentially-independent drafts by ~0.46 accepted tokens per step on average.

At inference DeepSeek-V3 either discards the MTP modules (and runs as a normal 671B MoE) or keeps them for speculative decoding. The main model is unchanged either way.

### Self-speculative decoding

Vanilla speculative decoding (Leviathan et al., 2023; Chen et al., 2023) requires two models: a cheap draft model and an expensive verifier. You run k draft tokens cheap, then one verifier forward pass accepts a prefix of them in parallel. Net speedup is the acceptance rate times k, minus overhead.

MTP makes the draft model free. The k-1 extra heads already predict `x_{t+2}, x_{t+3}, ...`. At decode step t:

1. Run the trunk once. Head 1 emits `x_{t+1}`. Heads 2..k emit draft tokens `d_2, ..., d_k`.
2. Feed `[x_{t+1}, d_2, ..., d_k]` into the next trunk forward pass as if they were real input.
3. Head 1's logits at each of those positions tell you whether it agrees. Accept the longest prefix where the drafted token matches head 1's argmax (greedy) or passes the acceptance criterion (sampling).

If all k-1 drafts are accepted you advanced k tokens in two forward passes instead of k. If none are accepted you get one token -- the baseline. The expected tokens per decode step is `1 + sum_{i=2..k} P(accept_i)`. This is the quantity the code in this lesson measures.

### Acceptance rate, in practice

The acceptance rate of draft i is the probability that `argmax(head_i) == argmax(head_1(h_{t+i-1}))`. It falls fast with depth:

- Depth 2: ~0.5-0.7 for trained MTP heads. The head-2 prediction is conditioned on t's hidden state; it is a reasonable forecast for a single step.
- Depth 3: ~0.3-0.5. Further out, accuracy drops; the trunk has to encode more distinct information.
- Depth 4+: Diminishing. This is why Gloeckle et al. recommended k=4 and DeepSeek-V3 used a single MTP module (k=2).

The acceptance rate times k is a rough cap on the speedup you can buy. Code models see higher acceptance than chat models because code has more local structure (closing brackets, keywords, repeated tokens).

### What MTP is not

- **Not parallel decoding.** You still run one trunk pass per accepted-block. MTP changes how much each pass contributes, not what a pass is.
- **Not Medusa or Hydra.** Medusa (Cai et al., 2024) adds heads on top of a frozen backbone after pre-training. MTP heads are trained jointly with the trunk from scratch, so the trunk's representations are shaped by the MTP loss.
- **Not lossless in the training-cost sense.** The Gloeckle et al. result is that MTP adds training signal without adding wall-clock time. That is because the heads are thin. DeepSeek-V3's sequential version does add meaningful training cost.

### Tree attention vs linear draft

When you have k-1 drafts you have a choice. The simple version -- the one in this lesson's code -- treats the drafts as a single candidate sequence and verifies them left to right. If draft 2 is wrong, drafts 3..k are discarded even if they would have been right after resampling.

Medusa-style tree attention (Cai et al., 2024) instead expands each draft slot into the top-m candidates, arranging them as a tree of length (k-1) with branching factor m. One trunk forward pass verifies every root-to-leaf path at once using an attention mask that isolates each path. Expected accepted length grows because the verifier now picks the best path, not just a single guess. Tree attention and MTP are orthogonal: DeepSeek-V3's serving stack combines them.

### Where the gain comes from

Think about what the trunk has to learn to make head 3 accurate. The hidden state `h_t` must encode, simultaneously:

- The identity of token t+1 (for head 1).
- The identity of token t+2 (for head 2).
- The identity of token t+3 (for head 3).

Features that are useful for three offsets get reinforced. Features useful only for the immediate next token get weaker (relatively) because they show up in only one of three loss terms. The result is a representation that captures longer-range structure at the same cost. This is why MTP helps code more than chat -- code has tighter local dependencies (a `(` at position t strongly predicts `)` at t+k for some k), and the parallel-futures loss lets the trunk commit to them.

### MTP as a side effect of curriculum

An easy way to reason about MTP: it is next-token prediction with an auxiliary "look ahead" loss. The extra heads are regularizers. You do not ship them to production unless you want speculative decoding. The main model stays exactly the shape you would have trained anyway. This is the property that makes MTP a safe auxiliary objective: worst case, you ignore the heads at inference and you have your original model back, trained slightly better.

## Build It

The code in `code/main.py` is a toy end-to-end simulation. Concretely:

1. A tiny character-level "transformer" (just an embedding table plus a single linear layer of context mixing -- enough to have non-trivial dynamics) trained on a short repeating corpus.
2. An MTP training loop that instantiates k heads (k=2 and k=3), each projecting the shared hidden state to the vocabulary.
3. An inference loop that uses head 1 autoregressively and heads 2..k as draft tokens, with greedy verification against head 1's next argmax.
4. A metric that reports acceptance rate per depth and effective tokens per decode step.

Stdlib only -- no torch, no numpy. The training is deliberately toy; the point is the MTP bookkeeping and the acceptance rate measurement, not SOTA perplexity.

The core training step, stripped of plumbing, is three lines:

```
hidden = trunk(x_t)
for i in range(1, k + 1):
    logits_i = head_i(hidden)
    loss += cross_entropy(logits_i, x_{t+i})
```

That is it. The trunk never knows there are multiple heads. Each head sees the same hidden state and is trained against a different future token. At inference you ignore heads 2..k unless you want drafts:

```
x_{t+1} = argmax(head_1(hidden))
drafts  = [argmax(head_i(hidden)) for i in 2..k]
```

Then the verification step: feed `[x_{t+1}, drafts]` back through the trunk and check which drafts match head 1's new argmax at each position. The longest matching prefix is accepted in one shot.

## Use It

Run `python code/main.py`. The script prints the trained model's next-token accuracy, then acceptance rate at depth 2 and depth 3, then the effective tokens-per-step estimate. Try increasing the number of training steps and the context length to see acceptance rates rise as the trunk learns richer representations.

Then vary k. With k=2 you buy at most 2x. With k=3 you buy at most 3x, but depth-3 acceptance is always lower than depth-2, so diminishing returns set in quickly.

## Ship It

This lesson produces `outputs/skill-mtp.md`. Given a production decoding setup (model family, serving stack, target workload), it recommends whether to train MTP heads, which variant (parallel vs sequential), and what acceptance-rate floor the resulting draft needs to hit before it beats vanilla speculative decoding with a separate draft model.

## Exercises

1. Compute the expected wall-clock speedup for k=4 with acceptance rates (0.6, 0.4, 0.25) at depths 2, 3, 4. Compare to the theoretical ceiling of 4x.

2. Gloeckle et al. report no gain on models under ~1B parameters. Why would MTP hurt small models? Think about the gradient interference between heads at low capacity.

3. Sequential MTP (DeepSeek-V3) gets higher acceptance than parallel MTP (Gloeckle). Why is that consistent with Hydra's result that sequentially-dependent draft heads beat sequentially-independent ones?

4. Modify the code in `main.py` to add a fourth MTP head. Measure acceptance at depth 4 on the same corpus. Report whether the effective-tokens-per-step metric improved.

5. A draft model trained from scratch (vanilla speculative decoding) typically has 0.5-0.7 acceptance on chat workloads. Where does the MTP self-speculation break even against it? Compare parameter cost of k-1 extra linear heads vs a 1B draft model.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| MTP | "Predict multiple tokens" | Train the model with loss over the next k future tokens, not just the next one |
| NTP | "Regular language modeling" | Standard next-token-prediction: one head, one future token, one loss per position |
| Parallel MTP | "Gloeckle-style heads" | k independent linear heads reading the same shared hidden state |
| Sequential MTP | "DeepSeek-V3 modules" | k chained mini-transformers where module i conditions on module i-1's prediction |
| Depth (k) | "How far ahead we predict" | Index of the future token: depth 1 is next token, depth k is k tokens ahead |
| Acceptance rate | "How often the draft is right" | P(drafted token at depth i matches the verifier's argmax at that position) |
| Self-speculative decoding | "Free speculation" | Use the same model's own MTP heads as the draft; no external draft model needed |
| Medusa | "Heads added after training" | Fine-tuned draft heads on a frozen backbone, k=4 typical |
| Hydra | "Sequential Medusa" | Medusa with sequentially-dependent draft heads; higher acceptance |
| Effective tokens/step | "Real-world speedup" | 1 + sum of per-depth acceptance rates; caps at k; what you actually measure on latency |

## Further Reading

- [Gloeckle et al., 2024 -- "Better & Faster Large Language Models via Multi-token Prediction"](https://arxiv.org/abs/2404.19737) -- the ICML 2024 paper that introduced parallel MTP heads with shared trunk
- [DeepSeek-AI, 2024 -- "DeepSeek-V3 Technical Report"](https://arxiv.org/abs/2412.19437) -- Section 2.2 specifies the sequential MTP module variant used to train the 671B MoE
- [Leviathan et al., 2023 -- "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) -- the original speculative decoding algorithm that MTP self-drafts plug into
- [Chen et al., 2023 -- "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) -- the parallel DeepMind formulation of speculative decoding
- [Cai et al., 2024 -- "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) -- post-hoc multi-head drafts on a frozen backbone
- [Ankner et al., 2024 -- "Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding"](https://arxiv.org/abs/2402.05109) -- sequentially-dependent draft heads, the idea DeepSeek-V3 adopted for sequential MTP
