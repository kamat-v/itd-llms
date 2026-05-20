"""
Experiment 2: hallucination on 15 questions x 3 context conditions.

For each model and each item, we run both:
  (a) greedy generation followed by case-insensitive substring match for
      the answer (the same accuracy convention as the executed notebook);
  (b) answer-only NLL of the correct answer in each of the three conditions
      (no / relevant / misleading), using -100 masking on prompt tokens.

Output: results/exp2_results.json

Runtime: roughly 4 hours on a single A100 80GB; mostly model loading.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from models import ALL_MODELS, INSTRUCT_MODELS, load_model, cleanup_model, clear_memory
from nll import compute_answer_nll


def make_prompt(context: str, question: str) -> str:
    if context:
        return f"{context}\n\nQuestion: {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def get_answer(model, tokenizer, context: str, question: str, max_new_tokens: int = 15) -> str:
    """Greedy generation of an answer continuation."""
    prompt = make_prompt(context, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    import torch
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text.split('\n')[0]


def is_correct(generated: str, answer: str) -> bool:
    return answer.lower() in generated.lower()


def run(data_path: Path, output_path: Path, models=None) -> None:
    models=models or ALL_MODELS
    with open(data_path) as f:
        items = json.load(f)
    print(f"Loaded {len(items)} hallucination items from {data_path}")

    all_per_model = {}
    for model_path, model_name, model_size in tqdm(models, desc="Models"):
        print(f"\n=== {model_name} ({model_size}B) ===")
        try:
            model, tokenizer = load_model(model_path)
            per_item = []
            for item in tqdm(items, desc="Questions", leave=False):
                # Accuracy via greedy generation
                ans_none = get_answer(model, tokenizer, "",                       item["question"])
                ans_rel  = get_answer(model, tokenizer, item["context_relevant"],   item["question"])
                ans_mis  = get_answer(model, tokenizer, item["context_misleading"], item["question"])
                # Answer-only NLL of the correct answer in each condition
                nll_none = compute_answer_nll(model, tokenizer,
                                              make_prompt("",                       item["question"]),
                                              item["correct_answer"])
                nll_rel  = compute_answer_nll(model, tokenizer,
                                              make_prompt(item["context_relevant"], item["question"]),
                                              item["correct_answer"])
                nll_mis  = compute_answer_nll(model, tokenizer,
                                              make_prompt(item["context_misleading"], item["question"]),
                                              item["correct_answer"])

                per_item.append({
                    "id": item["id"],
                    "question": item["question"],
                    "correct_answer": item["correct_answer"],
                    "generated_no_ctx":      ans_none,
                    "generated_relevant":    ans_rel,
                    "generated_misleading":  ans_mis,
                    "correct_no_ctx":      bool(is_correct(ans_none, item["correct_answer"])),
                    "correct_relevant":    bool(is_correct(ans_rel,  item["correct_answer"])),
                    "correct_misleading":  bool(is_correct(ans_mis,  item["correct_answer"])),
                    "nll_no_ctx":      float(nll_none),
                    "nll_relevant":    float(nll_rel),
                    "nll_misleading":  float(nll_mis),
                })

            # Per-model summary
            acc_none = float(np.mean([r['correct_no_ctx']     for r in per_item]))
            acc_rel  = float(np.mean([r['correct_relevant']   for r in per_item]))
            acc_mis  = float(np.mean([r['correct_misleading'] for r in per_item]))
            nll_none = float(np.nanmean([r['nll_no_ctx']     for r in per_item]))
            nll_rel  = float(np.nanmean([r['nll_relevant']   for r in per_item]))
            nll_mis  = float(np.nanmean([r['nll_misleading'] for r in per_item]))
            n_flips = int(sum(1 for r in per_item
                              if r['correct_relevant'] and not r['correct_misleading']))

            all_per_model[model_name] = {
                "size_b": float(model_size),
                "accuracy_no_ctx":      acc_none,
                "accuracy_relevant":    acc_rel,
                "accuracy_misleading":  acc_mis,
                "mean_nll_no_ctx":      nll_none,
                "mean_nll_relevant":    nll_rel,
                "mean_nll_misleading":  nll_mis,
                "n_flips": n_flips,
                "flip_rate": float(n_flips) / len(items),
                "per_item": per_item,
            }
            print(f"  acc no/rel/mis: {acc_none:.0%}/{acc_rel:.0%}/{acc_mis:.0%}  "
                  f"NLL: {nll_none:.3f}/{nll_rel:.3f}/{nll_mis:.3f}  "
                  f"flips: {n_flips}/{len(items)}")

            cleanup_model(model, tokenizer)
            with open(output_path, 'w') as f:
                json.dump({"per_model": all_per_model}, f, indent=2)

        except Exception as e:
            print(f"  ERROR on {model_name}: {e}")
            import traceback; traceback.print_exc()
            clear_memory()

    # Aggregate
    accs_none = [v['accuracy_no_ctx']     for v in all_per_model.values()]
    accs_rel  = [v['accuracy_relevant']   for v in all_per_model.values()]
    accs_mis  = [v['accuracy_misleading'] for v in all_per_model.values()]
    nlls_none = [v['mean_nll_no_ctx']     for v in all_per_model.values()]
    nlls_rel  = [v['mean_nll_relevant']   for v in all_per_model.values()]
    nlls_mis  = [v['mean_nll_misleading'] for v in all_per_model.values()]

    summary = {
        "n_models": len(all_per_model),
        "n_questions": len(items),
        "aggregate": {
            "mean_accuracy_no_ctx":      float(np.mean(accs_none)),
            "mean_accuracy_relevant":    float(np.mean(accs_rel)),
            "mean_accuracy_misleading":  float(np.mean(accs_mis)),
            "mean_nll_no_ctx":           float(np.mean(nlls_none)),
            "mean_nll_relevant":         float(np.mean(nlls_rel)),
            "mean_nll_misleading":       float(np.mean(nlls_mis)),
            "n_models_mis_below_no_ctx": int(sum(1 for v in all_per_model.values()
                                                 if v['accuracy_misleading'] < v['accuracy_no_ctx']
                                                 and v['accuracy_no_ctx'] > 0)),
            "n_models_with_any_flip":    int(sum(1 for v in all_per_model.values()
                                                 if v['n_flips'] >= 1)),
            "n_models_nll_mis_below_no": int(sum(1 for v in all_per_model.values()
                                                 if v['mean_nll_misleading'] < v['mean_nll_no_ctx'])),
        },
        "per_model": all_per_model,
    }
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {output_path}")
    agg = summary['aggregate']
    print(f"  acc no/rel/mis: {agg['mean_accuracy_no_ctx']:.0%}/"
          f"{agg['mean_accuracy_relevant']:.0%}/{agg['mean_accuracy_misleading']:.0%}")
    print(f"  NLL no/rel/mis: {agg['mean_nll_no_ctx']:.3f}/"
          f"{agg['mean_nll_relevant']:.3f}/{agg['mean_nll_misleading']:.3f} nats")
    print(f"  models with NLL_mis < NLL_no_ctx: "
          f"{agg['n_models_nll_mis_below_no']}/{summary['n_models']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("data/hallucination_triples.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/exp2_results.json"))
    parser.add_argument("--instruct-output", type=Path,
                        default=Path("results/exp2_instruct_results.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.data, args.output)
    run(args.data, args.instruct_output, models=INSTRUCT_MODELS)
