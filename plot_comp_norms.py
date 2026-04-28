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


def _save_multi_plot(x, ys: List[tuple[str, pd.Series]], title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    for label, series in ys:
        plt.plot(x, series, linewidth=1.8, label=label)
    plt.title(title)
    plt.xlabel("episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_conflicts(conflicts_log: Path, out_dir: Path) -> None:
    if not conflicts_log.exists():
        print(f"[WARN] conflicts log not found, skipping conflict plots: {conflicts_log}")
        return

    cdf = pd.read_csv(conflicts_log)
    _require_columns(
        cdf,
        [
            "episode",
            "conflicts_total",
            "tasks_total",
            "conflict_ratio",
            "winner_pickup",
            "winner_margin",
            "resolver_override",
            "resolver_override_rate",
            "avg_margin_win",
            "avg_margin_lose",
            "avg_margin_gap",
        ],
    )

    if "p_win_given_ego_action" not in cdf.columns:
        denom = cdf["conflicts_total"].replace(0, pd.NA)
        cdf["p_win_given_ego_action"] = (cdf["winner_pickup"] / denom).fillna(0.0)
    if "p_win_given_high_logit" not in cdf.columns:
        denom = cdf["conflicts_total"].replace(0, pd.NA)
        cdf["p_win_given_high_logit"] = (cdf["winner_margin"] / denom).fillna(0.0)

    x = cdf["episode"]

    _save_multi_plot(
        x,
        [
            ("conflicts_total", cdf["conflicts_total"]),
            ("tasks_total", cdf["tasks_total"]),
        ],
        "Conflicts Total vs Tasks Total",
        "count",
        out_dir / "conflicts_total_vs_tasks_total.png",
    )

    _save_plot(
        x,
        cdf["conflict_ratio"],
        "Conflict Ratio",
        "conflict_ratio",
        out_dir / "conflict_ratio.png",
    )

    _save_multi_plot(
        x,
        [
            ("winner_pickup", cdf["winner_pickup"]),
            ("winner_margin", cdf["winner_margin"]),
            ("resolver_override", cdf["resolver_override"]),
        ],
        "Winner Pickup vs Winner Margin vs Resolver Override",
        "count",
        out_dir / "winner_pickup_margin_override.png",
    )

    _save_plot(
        x,
        cdf["resolver_override_rate"],
        "Resolver Override Rate",
        "resolver_override_rate",
        out_dir / "resolver_override_rate.png",
    )

    _save_multi_plot(
        x,
        [
            ("P(win|ego_action)", cdf["p_win_given_ego_action"]),
            ("P(win|high_logit)", cdf["p_win_given_high_logit"]),
        ],
        "Conflict Win Probabilities",
        "probability",
        out_dir / "win_probability_diagnostics.png",
    )

    _save_multi_plot(
        x,
        [
            ("avg_margin_win", cdf["avg_margin_win"]),
            ("avg_margin_lose", cdf["avg_margin_lose"]),
            ("avg_margin_gap", cdf["avg_margin_gap"]),
        ],
        "Average Margins",
        "margin",
        out_dir / "avg_margins.png",
    )

    print(f"[OK] Wrote conflict plots to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot comp_norms.log metrics")
    parser.add_argument("--log", type=str, default="comp_norms.log", help="Path to comp_norms.log")
    parser.add_argument("--conflicts-log", type=str, default="conflicts.log", help="Path to conflicts.log")
    parser.add_argument("--out", type=str, default="comp_norms_plots", help="Output folder for plots")
    args = parser.parse_args()

    log_path = Path(args.log)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if log_path.exists():
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

        print(f"[OK] Wrote comp_norms plots to {out_dir}")
    else:
        print(f"[WARN] comp norms log not found, skipping comp plots: {log_path}")

    _plot_conflicts(Path(args.conflicts_log), out_dir)


if __name__ == "__main__":
    main()
