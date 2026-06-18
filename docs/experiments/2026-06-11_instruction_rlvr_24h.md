# 2026-06-11 Instruction and RLVR 24-Hour Queue

## Motivation

The previous SFT mixture contained 85% function calling and 15% GSM8K replay.
It produced useful internal tool-call scores but only 33-34% strict prompt
accuracy on official IFEval. The new stage targets instruction-following
recovery without discarding tool behavior.

## SFT v3 data

The deterministic 128,000-row mixture contains:

| Source | Share | Rows |
|---|---:|---:|
| xLAM family-isolated tool data | 35% | 44,800 |
| Tulu 3 Persona Instruction Following | 25% | 32,000 |
| No Robots | 15% | 19,200 |
| Synthetic verifiable constraints | 15% | 19,200 |
| GSM8K replay | 10% | 12,800 |

IFEval test prompts and responses are not included. Synthetic tasks cover JSON
schemas, exact list lengths, required and forbidden phrases, prefixes/suffixes,
sentence counts, and bounded word counts.

## Queue

1. Continue Qwen2.5-Coder-3B hard90 with v3 SFT for 12,000 steps.
2. Run 1,200 steps of four-sample group-relative RLVR on deterministic
   constraint rewards.
3. Continue Qwen3-4B hard77 with v3 SFT for 10,000 steps.
4. Continue Qwen2.5-3B hard77 with v3 SFT for 12,000 steps.

Every stage runs official IFEval and the internal xLAM, GSM8K, MMLU-Pro, and
WikiSQL regression bundle. Expected project GPU demand after acquisition is
approximately 26-32 hours.

## RLVR reward

Each response receives per-constraint binary components, their mean as shaped
reward, a success bonus when all constraints pass, and a capped excessive
length penalty. Advantages are normalized within four independently sampled
responses for the same prompt. Groups with zero reward variance are skipped.

The implementation logs reward mean/std, exact success, skipped-group rate,
completion length, gradient norm, and learning rate. This is a lightweight
GRPO-style RLVR experiment, not yet the production `verl` agent loop.

## Safety gates

- Do not promote a checkpoint if IFEval improves while xLAM, GSM8K, or
  MMLU-Pro regress by more than two points.
- Stop RLVR if skipped groups exceed 80%, completion length grows persistently,
  gradient values become non-finite, or verifier success diverges from manual
  audits.
- Official BFCL and tau2-bench remain required before making market-level Agent
  claims.
