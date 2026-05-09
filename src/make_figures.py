"""
Regenerate the two figures used in the paper from `results/*.json`.

Outputs:
  figures/figure_3_causal.pdf       (Figure 1 in paper)
  figures/figure_5_hallucination.pdf (Figure 2 in paper)

Run after `run_experiment_1.py` and `run_experiment_2.py` have completed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Publication-quality settings (kept identical to the executed notebook)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

COLORS = {'blue': '#648FFF', 'red': '#D62728', 'green': '#2E8B57', 'gray': '#8C8C8C'}


def short_name(name: str) -> str:
    return (name.replace('Pythia-', 'P-')
                .replace('GPT-2-', 'G-')
                .replace('Mistral-', 'M-')
                .replace('Qwen2.5-', 'Q-'))


def plot_exp1(results_path: Path, output_path: Path) -> None:
    with open(results_path) as f:
        data = json.load(f)
    rows = sorted(data['per_model'], key=lambda r: r['size_b'])
    models = [r['model'] for r in rows]
    causal = [r['causal_accuracy'] * 100 for r in rows]
    corr   = [r['correlational_accuracy'] * 100 for r in rows]
    gaps   = [r['gap'] * 100 for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, causal, width, label='Causal', color=COLORS['red'], alpha=0.8)
    ax1.bar(x + width/2, corr,   width, label='Correlational', color=COLORS['blue'], alpha=0.8)
    ax1.axhline(33.3, color='gray', linestyle='--', alpha=0.6, label='Chance (33%)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xlabel('Model (ordered by size)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([short_name(m) for m in models], rotation=45, ha='right', fontsize=8)
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 75)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.text(-0.12, 1.08, 'A', transform=ax1.transAxes, fontsize=14, fontweight='bold')

    bar_colors = [COLORS['green'] if g > 0 else COLORS['red'] for g in gaps]
    ax2.bar(x, gaps, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.axhline(data['mean_gap'] * 100, color=COLORS['blue'], linestyle='--',
                linewidth=2, label=f"Mean = {data['mean_gap']*100:+.1f}%")
    ax2.text(0.97, 0.97,
             f"{data['n_positive_gap']}/{data['n_models']} positive\np={data['sign_test_p']:.4f}",
             transform=ax2.transAxes, fontsize=9, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='gray', alpha=0.9))
    ax2.set_ylabel('Gap: Correlational - Causal (%)')
    ax2.set_xlabel('Model (ordered by size)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([short_name(m) for m in models], rotation=45, ha='right', fontsize=8)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(-0.12, 1.08, 'B', transform=ax2.transAxes, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_exp2(results_path: Path, output_path: Path) -> None:
    with open(results_path) as f:
        data = json.load(f)
    per_model = data['per_model']
    # Order rows by size_b
    rows = sorted(per_model.items(), key=lambda kv: kv[1]['size_b'])
    n = len(rows)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    cond_colors = [COLORS['gray'], COLORS['green'], COLORS['red']]

    for idx, (name, r) in enumerate(rows):
        ax = axes[idx]
        accs = [r['accuracy_no_ctx']*100, r['accuracy_relevant']*100, r['accuracy_misleading']*100]
        bars = ax.bar(['No', 'Rel', 'Mis'], accs, color=cond_colors,
                      alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.0f}%', ha='center', fontsize=8, fontweight='bold')
        if idx % ncols == 0:
            ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(f"{name}\n(flip {r['flip_rate']:.0%})", fontsize=9)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp1", type=Path,
                        default=Path("results/exp1_results.json"))
    parser.add_argument("--exp2", type=Path,
                        default=Path("results/exp2_results.json"))
    parser.add_argument("--out_dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.exp1.exists():
        plot_exp1(args.exp1, args.out_dir / "figure_3_causal.pdf")
    else:
        print(f"Skipping figure 1: {args.exp1} not found")
    if args.exp2.exists():
        plot_exp2(args.exp2, args.out_dir / "figure_5_hallucination.pdf")
    else:
        print(f"Skipping figure 2: {args.exp2} not found")
