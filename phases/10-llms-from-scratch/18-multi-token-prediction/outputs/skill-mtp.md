---
name: multi-token-prediction
description: Decide whether to add multi-token-prediction heads to a pre-training or post-training run, and whether to reuse them for self-speculative decoding at serving time.
version: 1.0.0
phase: 10
lesson: 18
tags: [mtp, multi-token-prediction, speculative-decoding, self-speculative, pre-training, deepseek-v3, gloeckle]
---

Given a model training plan (architecture family, parameter count, target domain, data volume, compute budget) and a serving plan (hardware, batch size, latency SLO, existing draft-model setup), recommend whether to train multi-token-prediction (MTP) heads and, if yes, which variant and how to use them at inference.

Produce:

1. Go / no-go on MTP training. Name the scale threshold: under ~1B parameters, Gloeckle et al. (2024) report no gain and occasional regressions; above ~7B the downstream gains on code and math are material. Reject MTP for small student models or short fine-tunes where the extra loss terms compete with the primary objective.
2. Variant choice. Parallel heads (Gloeckle et al., 2024) when the goal is cheap auxiliary signal with zero added training time and minimal inference cost. Sequential modules (DeepSeek-V3, 2024) when draft quality for self-speculative decoding matters more than training cost -- accept the extra parameters and per-step FLOPs in exchange for higher per-depth acceptance.
3. Depth (k). k=2 is the conservative default (DeepSeek-V3 ships one MTP module). k=4 is the Gloeckle paper's ceiling; past that, depth-k acceptance collapses and loss-interference gets real. Justify the chosen k against measured depth-2 accuracy on a pilot run.
4. Inference usage decision. Three options: (a) discard MTP heads and ship the trunk only -- pure auxiliary regularizer, zero runtime change; (b) reuse MTP heads for self-speculative decoding -- requires verifier forward pass and tree-attention or linear-verification kernel; (c) stack on top of an external draft-model speculative stack -- rarely a win, MTP heads are cheaper and usually suffice. Pick (a) unless the serving stack already has speculative-decoding plumbing.
5. Acceptance-rate floor. The speedup ceiling is `1 + sum_{i=2..k} P(accept_i)`, capped at k. Set a measured floor: if depth-2 acceptance is below ~0.4 on held-out decode traces, self-speculative decoding is not worth the verification overhead. Reject the whole plan if the pilot run cannot clear this bar.
6. Comparison with alternatives. Name the specific alternative: Medusa (heads added post-hoc on a frozen backbone, no training recipe changes), Hydra (sequentially-dependent Medusa drafts), EAGLE (hidden-state draft model), or vanilla two-model speculative decoding. Justify why MTP wins or loses against the leading alternative for this workload.

Hard rejects:
- MTP on a model under 1B parameters with a finite compute budget -- loss interference dominates.
- MTP without a joint training run -- post-hoc MTP heads are just Medusa; call them that and use Medusa's recipe instead.
- Self-speculative decoding when depth-2 acceptance is below ~0.3 on held-out decodes -- the verifier pass costs more than the accepted tokens save.
- Any recommendation that does not report measured acceptance at depth 2 and depth 3 from a pilot run on representative traffic.

Output: a one-page recommendation covering training variant, k, inference usage, and pilot-run metrics, with explicit acceptance-rate targets and a reject-if clause naming the metric that would flip the decision. End with a "revisit if..." paragraph naming the specific workload change (longer context, tighter latency budget, migration to an MoE backbone) that would re-open the choice.
