import argparse
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in comp_norms.log: {missing}")


def _save_plot(x, y, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, linewidth=1.8)
    plt.title(title)
    plt.xlabel("timesteps")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot comp_norms.log metrics")
    parser.add_argument("--log", type=str, default="comp_norms.log", help="Path to comp_norms.log")
    parser.add_argument("--out", type=str, default="comp_norms_plots", help="Output folder for plots")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    df = pd.read_csv(log_path)
    _require_columns(
        df,
        [
            "ts",
            "logit_base",
            "logit_comp",
            "ratio_comp_base",
            "ratio_comp_gap",
            "attn_entropy",
            "max_attn",
        ],
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = df["ts"]

    _save_plot(
        x,
        df["logit_base"],
        "Base logit (mean)",
        "logit_base",
        out_dir / "logit_base.png",
    )
    _save_plot(
        x,
        df["logit_comp"],
        "Competitor logit (mean)",
        "logit_comp",
        out_dir / "logit_comp.png",
    )

    _save_plot(
        x,
        df["ratio_comp_base"],
        "|logit_comp| / (|logit_base| + eps)",
        "ratio_comp_base",
        out_dir / "ratio_comp_base.png",
    )
    _save_plot(
        x,
        df["ratio_comp_gap"],
        "|logit_comp| / (top1-top2 gap + eps)",
        "ratio_comp_gap",
        out_dir / "ratio_comp_gap.png",
    )

    _save_plot(
        x,
        df["attn_entropy"],
        "Attention entropy (mean over tasks)",
        "attn_entropy",
        out_dir / "attn_entropy.png",
    )
    _save_plot(
        x,
        df["max_attn"],
        "Max attention weight (mean over tasks)",
        "max_attn",
        out_dir / "max_attn.png",
    )

    print(f"[OK] Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
