# Paper claim → result field map

For every numerical claim in the paper, this file points to the JSON field in `results/` that produces it. Use this as a reproducibility audit tool.

After running both experiments, the expected values are listed under "expected" below. They were produced by an A100 80GB run; minor float-level differences across hardware are normal and do not affect any claim.

## Notation

- `exp1` = `results/exp1_results.json`
- `exp2` = `results/exp2_results.json`
- "as percentage" means the JSON field is a fraction in [0,1]; multiply by 100 for the paper's percentage.

## Abstract

| Paper claim | Source | Expected |
|---|---|---|
| "12 base language models (70M-32B, five families)" | `len(exp1.per_model)` and `src/models.py` | 12 |
| "every model prefers the correlational answer" | `exp1.n_positive_gap` | 12 |
| "mean gap +21.8 pp" | `exp1.mean_gap × 100` | +21.8 |
| "p = 0.0002" | `exp1.sign_test_p` | 0.0002 |
| "7 of 12 models show accuracy below the no-context baseline" | `exp2.aggregate.n_models_mis_below_no_ctx` | 7 |
| "12 of 12 models the negative log-likelihood ... stays below the no-context baseline" | `exp2.aggregate.n_models_nll_mis_below_no` | 12 |
| "Mistral-7B, Mistral-Nemo-12B, Qwen2.5-32B" robust | `exp2.per_model[name].accuracy_misleading >= exp2.per_model[name].accuracy_relevant` for those three | True |

## Section 3.2 (correlational bias)

| Paper claim | Source | Expected |
|---|---|---|
| "every one of the 12 models scores higher" | `all(r.gap > 0 for r in exp1.per_model)` | True |
| "+21.8 ± 10.9 percentage points" | `exp1.mean_gap × 100` and `exp1.std_gap × 100` | +21.8, 10.9 |
| "sign test p = 0.0002" | `exp1.sign_test_p` | 0.0002 |
| Per-category range "+18% to +26%" | computed from per-pair correctness in `per_model[*]` (not pre-aggregated in current JSON) | 18% to 26% |

## Section 3.3 (hallucination — accuracy)

| Paper claim | Source | Expected |
|---|---|---|
| "mean accuracy 59% (no-ctx)" | `exp2.aggregate.mean_accuracy_no_ctx × 100` | 59% |
| "83% (relevant)" | `exp2.aggregate.mean_accuracy_relevant × 100` | 83% |
| "50% (misleading)" | `exp2.aggregate.mean_accuracy_misleading × 100` | 50% |
| "8 of 12 models flip at least one answer" | `exp2.aggregate.n_models_with_any_flip` | 8 |
| "7 of these score below their no-context baseline" | `exp2.aggregate.n_models_mis_below_no_ctx` | 7 |
| "Pythia-70M ... 0% across all conditions" | `exp2.per_model['Pythia-70M']` accuracies | all 0 |
| "20% (Qwen2.5-14B)" flip rate | `exp2.per_model['Qwen2.5-14B'].flip_rate` | 0.20 |
| "100% (OPT-350M)" flip rate | `exp2.per_model['OPT-350M'].flip_rate` | 1.00 |

## Section 3.3 (hallucination — NLL)

| Paper claim | Source | Expected |
|---|---|---|
| "mean NLL of the correct answer is 4.64 nats with no context" | `exp2.aggregate.mean_nll_no_ctx` | 4.64 |
| "2.55 nats with relevant context" | `exp2.aggregate.mean_nll_relevant` | 2.55 |
| "3.67 nats with misleading context" | `exp2.aggregate.mean_nll_misleading` | 3.67 |
| "still 0.98 nats below the no-context baseline" | `exp2.aggregate.mean_nll_no_ctx - exp2.aggregate.mean_nll_misleading` | 0.97 (rounds to 0.98) |
| "in 12 of 12 models, NLL of the correct answer under misleading context remains below the no-context value" | `exp2.aggregate.n_models_nll_mis_below_no` | 12 |

## Table 1 (NLL by model)

The 12 rows of Table 1 in the paper come from `exp2.per_model[<name>].mean_nll_*`. Per-model expected values (no_ctx / relevant / misleading):

| Model | No ctx | Relevant | Misleading |
|---|---|---|---|
| Pythia-70M | 12.60 | 9.19 | 9.41 |
| GPT-2-124M | 7.34 | 3.61 | 5.51 |
| OPT-350M | 6.46 | 2.47 | 5.52 |
| Pythia-410M | 5.09 | 2.78 | 4.70 |
| Pythia-1B | 4.80 | 2.40 | 4.03 |
| GPT-2-XL | 3.96 | 1.94 | 3.38 |
| Pythia-2.8B | 2.64 | 1.75 | 2.37 |
| Mistral-7B | 0.84 | 0.77 | 0.78 |
| Qwen2.5-7B | 3.94 | 1.69 | 3.09 |
| Mistral-Nemo-12B | 1.72 | 1.50 | 1.36 |
| Qwen2.5-14B | 2.78 | 1.03 | 2.05 |
| Qwen2.5-32B | 3.55 | 1.47 | 1.82 |

## Appendix A Table 2 (per-model causal/correlational accuracy)

The 12 rows come from `exp1.per_model`. Expected (causal / correlational / gap pp):

| Model | Causal | Corr | Gap |
|---|---|---|---|
| Pythia-70M | 44 | 48 | +4 |
| GPT-2-124M | 44 | 54 | +10 |
| OPT-350M | 28 | 46 | +18 |
| Pythia-410M | 26 | 54 | +28 |
| Pythia-1B | 24 | 58 | +34 |
| GPT-2-XL | 30 | 64 | +34 |
| Pythia-2.8B | 30 | 64 | +34 |
| Mistral-7B | 30 | 68 | +38 |
| Qwen2.5-7B | 36 | 58 | +22 |
| Mistral-Nemo-12B | 46 | 60 | +14 |
| Qwen2.5-14B | 50 | 64 | +14 |
| Qwen2.5-32B | 50 | 62 | +12 |

## Appendix B Table 3 (per-model hallucination accuracy and flips)

The 12 rows come from `exp2.per_model`. Expected per row: `accuracy_no_ctx`, `accuracy_relevant`, `accuracy_misleading`, `n_flips/15`.

## Notes on reproducibility

- Floating-point differences at the 0.01-nat level across hardware/torch versions are expected and do not change any claim.
- Greedy generation is deterministic; the accuracy numbers should reproduce exactly given the same model weights and tokenizer.
- The full-sequence NLL ranking in Experiment 1 is also deterministic.
- Some models load with bfloat16 + auto device placement; very small numerical differences may accumulate but do not flip any per-model gap sign.
