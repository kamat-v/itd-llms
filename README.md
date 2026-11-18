# An Information-Theoretic Diagnostic for Hallucination and Correlational Bias in LLMs

Reproduction code and benchmarks for the paper of the same title, accepted at the 19th International Conference on Agents and Artificial Intelligence (ICAART 2027).

**Authors:** Uday Kamath, Vikram Kamat

## What this repository contains

- **`data/`** — the two benchmarks used by the paper, as JSON files with explicit schemas:
  - `causal_pairs.json`: 50 matched causal/correlational question pairs across five categories.
  - `hallucination_triples.json`: 15 factual questions, each in three context conditions (no/relevant/misleading).
- **`src/`** — standalone Python scripts that run the experiments and regenerate the figures.
- **`paper/claims.md`** — a one-to-one map from every numerical claim in the paper to the JSON field that produces it.

The repository is intentionally minimal: no Python package, no CI, no tests. It is set up so you can clone it, install the requirements, and run two scripts to reproduce the paper's tables and figures.

## Quickstart

```bash
git clone https://github.com/kamat-v/itd-llms.git
cd itd-llms
pip install -r requirements.txt

# Experiment 1: correlational bias (12 base + 4 instruct models x 50 matched pairs)
python src/run_experiment_1.py

# Experiment 2: hallucination (12 base + 4 instruct models x 15 questions x 3 conditions)
python src/run_experiment_2.py

# Regenerate figures from the JSON results
python src/make_figures.py
```

Outputs land in `results/` (JSON) and `figures/` (PDF).

## Hardware and runtime

The experiments were run on a single NVIDIA A100 (40GB). Approximate wall-clock:

- Experiment 1: ~60-90 minutes, dominated by the larger Qwen models.
- Experiment 2: ~4 hours, dominated by model loading.

Smaller GPUs can run the experiments by editing `ALL_MODELS` or `INSTRUCT_MODELS` in `src/models.py` to remove larger models.

The first run downloads ~120 GB of model weights from HuggingFace into the local cache. Subsequent runs read from cache.

## Models

All 12 base models in `ALL_MODELS` and all 4 instruct models in `INSTRUCT_MODELS` are ungated on HuggingFace; no login or access request is required. They span a 457x parameter range across five families (Pythia, GPT-2, OPT, Mistral, Qwen). See `src/models.py` for the exact list.

## Reproducing specific paper claims

`paper/claims.md` maps each numerical claim in the paper to the corresponding JSON field in `results/`. For instance, the paper's "+21.8 pp" mean gap is `results/exp1_results.json -> mean_gap` (multiplied by 100).

## Extending the benchmarks

Both JSON files have stable schemas documented in `data/README.md`. To run the diagnostic on additional models, append entries to `ALL_MODELS` or `INSTRUCT_MODELS` in `src/models.py`. To run on additional questions, append entries to either JSON file using the existing schema; the scripts treat the lists as data and do not assume a fixed length.

## Citation

If you use this code or the benchmarks, please cite:

```bibtex
@inproceedings{kk2027diagnostic,
  title     = {An Information-Theoretic Diagnostic for Hallucination and Correlational Bias in {LLMs}},
  author    = {Kamath, Uday and Kamat, Vikram},
  booktitle = {Proceedings of the 19th International Conference on Agents and Artificial Intelligence (ICAART)},
  publisher = {SciTePress},
  year      = {2027}
}
```

Page numbers and DOI will be added once the proceedings are published.

## License

MIT for the code; the benchmark JSON files are released under the same license. See `LICENSE`.
