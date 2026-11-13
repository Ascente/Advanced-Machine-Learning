"""Export local training/validation curves from metrics.jsonl."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def plot_metrics(metrics_path, output_dir):
    rows = [json.loads(line) for line in Path(metrics_path).read_text().splitlines() if line.strip()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, metric, label in zip(axes, ("loss", "ppl", "bpb"), ("Cross-entropy loss", "Perplexity", "Bits per byte")):
        for split, name in (("train", "Train (training batches)"), ("validation", "Validation (full split)")):
            selected = [row for row in rows if row["split"] == split and not row.get("final")]
            if selected:
                axis.plot([row["tokens_seen"] for row in selected], [row[metric] for row in selected], label=name)
        full = [row for row in rows if row.get("final")]
        if full:
            axis.scatter([row["tokens_seen"] for row in full], [row[metric] for row in full], marker="*", s=70,
                         label="Best checkpoint (full validation)")
        axis.set(xlabel="Training tokens seen", ylabel=label)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "curves.png", dpi=160)
    fig.savefig(output_dir / "curves.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_metrics(args.metrics, args.output_dir)


if __name__ == "__main__":
    main()
