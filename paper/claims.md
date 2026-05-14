# Paper claim → result field map

For every numerical claim in the paper, this file points to the JSON field in `results/` that produces it. Use this as a reproducibility audit tool.

After running both experiments, the expected values are listed under "expected" below. They were produced by an A100 80GB run; minor float-level differences across hardware are normal and do not affect any claim.

## Notation

- `exp1` = `results/exp1_results.json` (12 base models)
- `exp2` = `results/exp2_results.json` (12 base models)
- `exp1_instruct` = `results/exp1_instruct_results.json` (4 instruct models)
- `exp2_instruct` = `results/exp2_instruct_results.json` (4 instruct models)

## Abstract

| Paper claim | Source | Expected |
|---|---|---|
| "12 base language models (70M-32B, five families)" | `len(exp1.per_model)` and `src/models.py` | 12 |
| "every model prefers the correlational answer" | `exp1.n_positive_gap` | 12 |
| "mean gap +22.2 pp" | `exp1.mean_gap × 100` | +22.2 |
| "p = 0.0002" | `exp1.sign_test_p` | 0.0002 |
| "8 of 12 models show accuracy below the no-context baseline" | `exp2.aggregate.n_models_mis_below_no_ctx` | 8 |
| "in 12 of 12 models the negative log-likelihood ... stays below the no-context baseline" | `exp2.aggregate.n_models_nll_mis_below_no` | 12 |
| "Robustness ... only in two models (Mistral-7B, Qwen2.5-32B)" | per-model: `accuracy_misleading >= accuracy_relevant AND n_flips == 0` | 2 of 12 |
| "instruction tuning attenuates the correlational gap" | `exp1_instruct.mean_gap` vs paired base | +10.5 vs +21.5 |
| "NLL signature persists in three of four instruct models" | `exp2_instruct.aggregate.n_models_nll_mis_below_no` | 3 of 4 |

## Section 3.2 (correlational bias, base)

| Paper claim | Source | Expected |
|---|---|---|
| "+22.2 ± 10.8 pp" | `exp1.mean_gap × 100`, `exp1.std_gap × 100` (population SD) | +22.2, 10.8 |
| "sign test p = 0.0002" | `exp1.sign_test_p` | 0.0002 |
| Per-category range "+18% to +26%" | computed from per-pair correctness | 18% to 26% |

## Section 3.3 (hallucination — accuracy, base)

| Paper claim | Source | Expected |
|---|---|---|
| "mean accuracy 60% (no-ctx)" | `exp2.aggregate.mean_accuracy_no_ctx` | 0.60 |
| "83% (relevant)" | `exp2.aggregate.mean_accuracy_relevant` | 0.83 |
| "50% (misleading)" | `exp2.aggregate.mean_accuracy_misleading` | 0.50 |
| "9 of 12 models flip at least one answer" | `exp2.aggregate.n_models_with_any_flip` | 9 |
| "8 of these score below their no-context baseline" | `exp2.aggregate.n_models_mis_below_no_ctx` | 8 |

## Section 3.3 (hallucination — NLL, base)

| Paper claim | Source | Expected |
|---|---|---|
| "4.61 nats with no context" | `exp2.aggregate.mean_nll_no_ctx` | 4.61 |
| "2.53 nats with relevant context" | `exp2.aggregate.mean_nll_relevant` | 2.53 |
| "3.65 nats with misleading context" | `exp2.aggregate.mean_nll_misleading` | 3.65 |
| "still 0.95 nats below the no-context baseline" | `mean_nll_no_ctx - mean_nll_misleading` | 0.95 |
| "in 12 of 12 models, NLL ... remains below the no-context value" | `exp2.aggregate.n_models_nll_mis_below_no` | 12 |

## Table 1 (NLL by model, base)

Per-model expected (no_ctx / relevant / misleading), nats:

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
| Mistral-Nemo-12B | 1.27 | 1.23 | 1.16 |
| Qwen2.5-14B | 2.78 | 1.03 | 2.05 |
| Qwen2.5-32B | 3.55 | 1.47 | 1.82 |

Note: Mistral-Nemo-12B values use `mistralai/Mistral-Nemo-Base-2407` (the base variant, not Instruct).

## Appendix A Table 2 (per-model causal/correlational accuracy, base)

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
| Mistral-Nemo-12B | 42 | 60 | +18 |
| Qwen2.5-14B | 50 | 64 | +14 |
| Qwen2.5-32B | 50 | 62 | +12 |

## Appendix B Table 3 (per-model hallucination, base)

| Model | No | Rel | Mis | Flip |
|---|---|---|---|---|
| Pythia-70M | 0% | 0% | 0% | 0/15 |
| GPT-2-124M | 7% | 60% | 13% | 7/15 |
| OPT-350M | 20% | 100% | 0% | 15/15 |
| Pythia-410M | 33% | 80% | 13% | 10/15 |
| Pythia-1B | 40% | 80% | 13% | 10/15 |
| GPT-2-XL | 60% | 87% | 53% | 6/15 |
| Pythia-2.8B | 87% | 100% | 67% | 5/15 |
| Mistral-7B | 100% | 93% | 100% | 0/15 |
| Qwen2.5-7B | 93% | 100% | 67% | 5/15 |
| Mistral-Nemo-12B | 100% | 100% | 93% | 1/15 |
| Qwen2.5-14B | 93% | 100% | 80% | 3/15 |
| Qwen2.5-32B | 87% | 100% | 100% | 0/15 |

## Appendix C (instruction-tuned comparison)

Exp 1 (instruct):

| Model | Causal | Corr | Gap (instruct) | Gap (base) |
|---|---|---|---|---|
| Mistral-7B-Instruct | 44% | 66% | +22 | +38 |
| Qwen2.5-7B-Instruct | 54% | 58% | +4 | +22 |
| Qwen2.5-14B-Instruct | 50% | 60% | +10 | +14 |
| Qwen2.5-32B-Instruct | 58% | 64% | +6 | +12 |
| Mean | | | +10.5 | +21.5 |

Sign test on 4-instruct sample: 4/4 positive, p = 0.0625 (the main statistical claim relies on the 12-base sign test, p = 0.0002).

Exp 2 (instruct, NLL no→mis):

| Model | No / Rel / Mis acc | Flip | NLL no→mis | Pattern |
|---|---|---|---|---|
| Mistral-7B-Instruct | 100/100/93% | 1/15 | 1.85→1.62 | preserved |
| Qwen2.5-7B-Instruct | 93/100/93% | 1/15 | 5.24→2.74 | preserved |
| Qwen2.5-14B-Instruct | 100/93/87% | 2/15 | 3.38→3.66 | **broken** |
| Qwen2.5-32B-Instruct | 100/100/93% | 1/15 | 4.82→3.54 | preserved |

The 4-model instruct run uses the same plain prompt format as the base run (no chat templates) so instruction tuning is the only variable that changes between paired runs.

## Appendix D (CLADDER pilot)

The CLADDER pilot is methodological (it explains why we did not use CLADDER for the main experiment) rather than a results claim. The numbers reported (parseability 52-71%, effective accuracy 16-30% with the strongest prompting strategy) come from a one-off pilot on 100 CLADDER questions using three models (Pythia-410M, GPT-2-XL, Qwen2.5-14B). The pilot is documented in the paper appendix; the full pilot data is not included in the released `results/` JSON files but can be regenerated on request.

## Reproducibility notes

- Float-level differences (~0.01 nat) across hardware/torch versions are expected and do not change any claim.
- Greedy generation and full-sequence NLL ranking are deterministic given the same model weights and tokenizer.
- Models load with bfloat16 + auto device placement.
