"""
Experiment 1: correlational bias on 50 matched causal/correlational pairs.

For each model and each pair, score every option of both questions by
full-sequence NLL of "Question: {q}\\n\\nAnswer: {opt}" and pick the
lowest-NLL option as the model's prediction. Aggregate per-model accuracy
on the causal and correlational versions and the gap between them.

Output: results/exp1_results.json

Runtime: roughly 2-3 hours on a single A100 80GB; dominated by Qwen2.5-32B
loading and the larger models' inference.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from tqdm.auto import tqdm

from models import ALL_MODELS, load_model, cleanup_model, clear_memory
from nll import compute_full_sequence_nll


def eval_mc(model, tokenizer, question, options):
    """Return index of the option with lowest full-sequence NLL."""
    nlls = []
    for opt in options:
        text = f"Question: {question}\n\nAnswer: {opt}"
        nlls.append(compute_full_sequence_nll(model, tokenizer, text))
    return int(np.argmin(nlls))


def run(data_path: Path, output_path: Path) -> None:
    with open(data_path) as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} matched pairs from {data_path}")

    results = []
    for model_path, model_name, model_size in tqdm(ALL_MODELS, desc="Models"):
        print(f"\n=== {model_name} ({model_size}B) ===")
        try:
            model, tokenizer = load_model(model_path)
            causal_correct = 0
            corr_correct = 0
            for i, pair in enumerate(tqdm(pairs, desc="Pairs", leave=False)):
                if eval_mc(model, tokenizer,
                           pair["causal"]["question"],
                           pair["causal"]["options"]) == pair["causal"]["correct_index"]:
                    causal_correct += 1
                if eval_mc(model, tokenizer,
                           pair["correlational"]["question"],
                           pair["correlational"]["options"]) == pair["correlational"]["correct_index"]:
                    corr_correct += 1

            causal_acc = causal_correct / len(pairs)
            corr_acc = corr_correct / len(pairs)
            results.append({
                "model": model_name,
                "size_b": float(model_size),
                "causal_accuracy": float(causal_acc),
                "correlational_accuracy": float(corr_acc),
                "gap": float(corr_acc - causal_acc),
            })
            print(f"  causal={causal_acc:.0%}  corr={corr_acc:.0%}  "
                  f"gap={corr_acc - causal_acc:+.0%}")

            cleanup_model(model, tokenizer)
            # Write intermediate results after each model in case of failure
            with open(output_path, 'w') as f:
                json.dump({"per_model": results}, f, indent=2)

        except Exception as e:
            print(f"  ERROR on {model_name}: {e}")
            import traceback; traceback.print_exc()
            clear_memory()

    # Aggregate statistics
    n_positive = sum(1 for r in results if r['gap'] > 0)
    sign_test_p = binomtest(n_positive, len(results), 0.5, alternative='greater').pvalue
    mean_gap = float(np.mean([r['gap'] for r in results]))
    std_gap = float(np.std([r['gap'] for r in results]))

    summary = {
        "n_models": len(results),
        "n_pairs": len(pairs),
        "n_positive_gap": int(n_positive),
        "sign_test_p": float(sign_test_p),
        "mean_gap": mean_gap,
        "std_gap": std_gap,
        "per_model": results,
    }
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {output_path}")
    print(f"  {n_positive}/{len(results)} models with positive gap, "
          f"mean {mean_gap*100:+.1f} pp, sign-test p={sign_test_p:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("data/causal_pairs.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/exp1_results.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.data, args.output)
