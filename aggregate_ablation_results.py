from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


@dataclass
class SummaryStats:
    ts_best: Optional[int]
    ts_last: Optional[int]
    best: float
    best_std: float
    last: float
    last_std: float
    last_k_mean: float
    last_k_std: float
    auc: float


TRAINING_METRICS = ["rew", "mdl", "wait", "comp", "noop", "cne_fr", "cne_mn"]
EVAL_METRICS = ["rew", "mdl", "wait", "comp", "noop", "cne_fr", "cne_mn"]
TRAIN_OUTPUT_METRICS = [
    "ep_rew_mean",
    "approx_kl",
    "clip_fraction",
    "entropy_loss",
    "explained_variance",
    "value_loss",
]


def _read_conf(path: Path) -> OmegaConf:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return OmegaConf.load(str(path))


def _ensure_list(value, n: int) -> List:
    if OmegaConf.is_list(value):
        value = list(value)
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"Expected list of length {n}, got {len(value)}")
        return list(value)
    return [value] * n


def _extract_exp_params(conf: OmegaConf, model_dirs: List[Path]) -> List[Dict[str, object]]:
    n = len(model_dirs)
    if bool(conf.read_params_from_yaml):
        out = []
        for model_dir in model_dirs:
            yaml_path = model_dir / "configs" / "rp_gnn.yaml"
            if not yaml_path.exists():
                raise FileNotFoundError(f"Missing config in {model_dir}: {yaml_path}")
            cfg = OmegaConf.load(str(yaml_path))
            out.append({
                "use_xy_pickup": bool(getattr(cfg.features, "use_xy_pickup", False)),
                "normalize_features": bool(getattr(cfg.features, "normalize_features", False)),
                "logit_temperature": float(getattr(cfg.ppo.policy_kwargs, "logit_temperature", 0.0)),
                "noop_init": float(getattr(cfg.ppo.policy_kwargs, "noop_init", 0.0)),
                "freeze_noop_logit": bool(getattr(cfg.ppo.policy_kwargs, "freeze_noop_logit", False)),
            })
        return out

    exp_params = conf.get("exp_params")
    if not exp_params:
        raise ValueError("exp_params must be provided when read_params_from_yaml is false")

    use_xy_pickup = _ensure_list(exp_params.get("use_xy_pickup"), n)
    normalize_features = _ensure_list(exp_params.get("normalize_features"), n)
    logit_temperature = _ensure_list(exp_params.get("logit_temperature"), n)
    noop_init = _ensure_list(exp_params.get("noop_init"), n)
    freeze_noop_logit = _ensure_list(exp_params.get("freeze_noop_logit"), n)

    out = []
    for i in range(n):
        out.append({
            "use_xy_pickup": bool(use_xy_pickup[i]),
            "normalize_features": bool(normalize_features[i]),
            "logit_temperature": float(logit_temperature[i]),
            "noop_init": float(noop_init[i]),
            "freeze_noop_logit": bool(freeze_noop_logit[i]),
        })
    return out


def _parse_metrics_log(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    header_line = None
    for line in lines:
        if line.lower().startswith("pol"):
            header_line = line
            break
    if header_line is None:
        raise ValueError(f"Could not find header in {path}")

    header_tokens = header_line.replace("|", " ").split()
    records = []
    for line in lines:
        if line.startswith("pol") or line.startswith("#"):
            continue
        if "|" not in line:
            continue
        tokens = line.replace("|", " ").split()
        if len(tokens) != len(header_tokens):
            continue
        record = dict(zip(header_tokens, tokens))
        records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    for col in df.columns:
        if col in {"pol"}:
            continue
        if col in {"seed", "ts"}:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["ts"]).copy()
    df["ts"] = df["ts"].astype(int)
    return df


def _parse_train_output(path: Path) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    current: Dict[str, float] = {}

    sep_re = re.compile(r"^-{3,}")
    kv_re = re.compile(r"\|\s+([^|]+?)\s+\|\s+([^|]+?)\s+\|")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if sep_re.match(line):
                if "total_timesteps" in current:
                    records.append(current)
                current = {}
                continue
            match = kv_re.search(line)
            if not match:
                continue
            key = match.group(1).strip()
            value = match.group(2).strip()
            try:
                val = float(value)
            except ValueError:
                continue
            current[key] = val

    if "total_timesteps" in current:
        records.append(current)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.rename(columns={"total_timesteps": "ts"})
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ts"]).copy()
    df["ts"] = df["ts"].astype(int)
    return df


def _find_first(path: Path, pattern: str) -> Optional[Path]:
    matches = sorted(path.glob(pattern))
    if matches:
        return matches[0]
    matches = sorted(path.rglob(pattern))
    if matches:
        return matches[0]
    return None


def _find_first_nonempty(
    path: Path,
    pattern: str,
    parser,
) -> tuple[Optional[Path], pd.DataFrame]:
    matches = list(path.glob(pattern))
    if not matches:
        matches = list(path.rglob(pattern))
    if not matches:
        return None, pd.DataFrame()
    matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in matches:
        try:
            df = parser(candidate)
        except Exception:
            continue
        if not df.empty:
            return candidate, df
    return matches[0], pd.DataFrame()


def _collect_eval_logs(model_dir: Path) -> Dict[str, Path]:
    eval_root = model_dir / "eval_results"
    if not eval_root.exists():
        return {}

    out = {}
    for mode in ("deterministic", "stochastic"):
        candidates = [p for p in eval_root.iterdir() if p.is_dir() and p.name.endswith(mode)]
        if not candidates:
            continue
        candidates.sort()
        log_path = candidates[0] / "evaluation_metrics.log"
        if log_path.exists():
            out[mode] = log_path
    return out


def _reference_ts_sets(model_dirs: List[Path]) -> List[List[int]]:
    ts_sets = []
    for model_dir in model_dirs:
        for log_path in _collect_eval_logs(model_dir).values():
            df = _parse_metrics_log(log_path)
            if df.empty:
                continue
            ts_sets.append(sorted(df["ts"].unique().tolist()))
    return ts_sets


def _reference_ts_sets_training(model_dirs: List[Path]) -> List[List[int]]:
    ts_sets = []
    for model_dir in model_dirs:
        _, train_output_df = _find_first_nonempty(model_dir, "train_output*.txt", _parse_train_output)
        if not train_output_df.empty:
            ts_sets.append(sorted(train_output_df["ts"].unique().tolist()))
        _, training_df = _find_first_nonempty(model_dir, "training_metrics*.log", _parse_metrics_log)
        if not training_df.empty:
            ts_sets.append(sorted(training_df["ts"].unique().tolist()))
    return ts_sets


def _pick_reference_ts(ts_sets: List[List[int]]) -> List[int]:
    if not ts_sets:
        raise ValueError("No evaluation logs found to determine reference timesteps.")
    ts_sets = sorted(ts_sets, key=lambda x: (len(x), x[-1] if x else -1))
    return ts_sets[0]


def _ts_step(ts_list: List[int]) -> Optional[int]:
    if len(ts_list) < 2:
        return None
    diffs = np.diff(sorted(ts_list))
    if len(diffs) == 0:
        return None
    values, counts = np.unique(diffs, return_counts=True)
    return int(values[np.argmax(counts)])


def _aggregate_by_ts(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby("ts")[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "ts" if c[0] == "ts" else f"{c[0]}_{c[1]}" for c in agg.columns.to_flat_index()
    ]
    return agg


def _map_ts_to_available(ts_ref: List[int], ts_available: List[int]) -> List[int]:
    if not ts_ref or not ts_available:
        return []
    available_sorted = sorted(ts_available)
    mapped = []
    for t in ts_ref:
        candidates = [x for x in available_sorted if x <= t]
        if candidates:
            mapped.append(candidates[-1])
    return sorted(set(mapped))


def _summary_for_metric(
    agg: pd.DataFrame,
    metric: str,
    ts_ref: List[int],
    k_eval: int,
    best_ts: Optional[int],
    compute_auc: bool,
    allow_floor_ts: bool = False,
) -> SummaryStats:
    if agg.empty:
        return SummaryStats(None, None, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    agg_map = agg.set_index("ts")
    if allow_floor_ts:
        ts_available = _map_ts_to_available(ts_ref, list(agg_map.index))
    else:
        ts_available = [t for t in ts_ref if t in agg_map.index]
    if not ts_available:
        return SummaryStats(None, None, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    ts_last = ts_available[-1]
    last = float(agg_map.loc[ts_last, mean_col])
    last_std = float(agg_map.loc[ts_last, std_col]) if std_col in agg_map else np.nan

    if best_ts is None or best_ts not in agg_map.index:
        ts_best = None
        best = np.nan
        best_std = np.nan
    else:
        ts_best = best_ts
        best = float(agg_map.loc[ts_best, mean_col])
        best_std = float(agg_map.loc[ts_best, std_col]) if std_col in agg_map else np.nan

    k = max(1, min(k_eval, len(ts_available)))
    last_k_ts = ts_available[-k:]
    last_k_vals = agg_map.loc[last_k_ts, mean_col].astype(float).values
    last_k_mean = float(np.mean(last_k_vals))
    last_k_std = float(np.std(last_k_vals, ddof=0))

    if compute_auc:
        ts_vals = np.array(ts_available, dtype=float)
        metric_vals = agg_map.loc[ts_available, mean_col].astype(float).values
        auc = float(np.trapezoid(metric_vals, ts_vals))
    else:
        auc = np.nan

    return SummaryStats(ts_best, ts_last, best, best_std, last, last_std, last_k_mean, last_k_std, auc)


def _best_ts_from_reward(
    agg: pd.DataFrame,
    ts_ref: List[int],
    best_metric: str = "rew",
    allow_floor_ts: bool = False,
) -> Optional[int]:
    if agg.empty:
        return None
    reward_col = f"{best_metric}_mean"
    agg_map = agg.set_index("ts")
    if allow_floor_ts:
        ts_available = _map_ts_to_available(ts_ref, list(agg_map.index))
    else:
        ts_available = [t for t in ts_ref if t in agg_map.index]
    if not ts_available:
        return None
    if reward_col not in agg_map.columns:
        return None
    values = agg_map.loc[ts_available, reward_col].astype(float)
    best_idx = int(values.idxmax())
    return best_idx


def _summarize_metrics(
    agg: pd.DataFrame,
    metrics: List[str],
    ts_ref: List[int],
    k_eval: int,
    best_metric: str = "rew",
    allow_floor_ts: bool = False,
) -> Dict[str, SummaryStats]:
    best_ts = _best_ts_from_reward(
        agg,
        ts_ref,
        best_metric=best_metric,
        allow_floor_ts=allow_floor_ts,
    )
    out: Dict[str, SummaryStats] = {}
    auc_metrics = {"rew", "mdl", "wait", "comp"}
    for metric in metrics:
        out[metric] = _summary_for_metric(
            agg,
            metric,
            ts_ref,
            k_eval,
            best_ts,
            metric in auc_metrics,
            allow_floor_ts=allow_floor_ts,
        )
    return out


def _write_summary_log(
    path: Path,
    rows: List[Dict[str, object]],
    ts_ref: List[int],
    ts_step: Optional[int],
) -> None:
    ts_first = ts_ref[0] if ts_ref else None
    ts_last = ts_ref[-1] if ts_ref else None

    with path.open("w", encoding="utf-8") as f:
        f.write(f"Ablation aggregation log ({datetime.now().isoformat(timespec='seconds')})\n")
        f.write(f"Reference timesteps: count={len(ts_ref)} first={ts_first} last={ts_last} step={ts_step}\n")
        f.write("\n")

        for row in rows:
            f.write(f"Model: {row['model_id']} | Eval: {row['eval_mode']}\n")
            f.write(
                "Params: use_xy_pickup={use_xy_pickup}, normalize_features={normalize_features}, "
                "logit_temperature={logit_temperature}, noop_init={noop_init}, freeze_noop_logit={freeze_noop_logit}\n".format(
                    **row
                )
            )
            f.write(f"ts_first={row['ts_first']} ts_last={row['ts_last']} ts_step={row['ts_step']}\n")
            f.write("Metrics:\n")

            for metric in row["metrics"]:
                stats: SummaryStats = row["metrics"][metric]
                f.write(
                    f"  {metric}: best={stats.best:.4f}±{stats.best_std:.4f} "
                    f"(ts={stats.ts_best}), last={stats.last:.4f}±{stats.last_std:.4f} "
                    f"(ts={stats.ts_last}), last_k_mean={stats.last_k_mean:.4f}±{stats.last_k_std:.4f}, "
                    f"auc={stats.auc:.4f}\n"
                )
            f.write("\n")


def main() -> None:
    conf = _read_conf(Path("ablation_conf.yaml"))
    model_dirs = [Path(p).expanduser().resolve() for p in conf.model_dirs]
    k_eval = int(conf.get("k_eval", 5))

    exp_params_list = _extract_exp_params(conf, model_dirs)

    training_only = bool(conf.get("training_only", False))
    if training_only:
        ts_sets = _reference_ts_sets_training(model_dirs)
        ts_ref = _pick_reference_ts(ts_sets) if ts_sets else []
        ts_step = _ts_step(ts_ref) if ts_ref else None
    else:
        ts_sets = _reference_ts_sets(model_dirs)
        ts_ref = _pick_reference_ts(ts_sets)
        ts_step = _ts_step(ts_ref)

    output_dir = Path(conf.get("output_dir", "ablation_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / conf.get("output_file", "ablation_aggregated.log")
    output_csv = output_dir / conf.get("output_csv", "ablation_aggregated.csv")

    rows: List[Dict[str, object]] = []

    for model_dir, exp_params in zip(model_dirs, exp_params_list):
        model_id = model_dir.name

        train_output, train_output_df = _find_first_nonempty(
            model_dir, "train_output*.txt", _parse_train_output
        )
        training_metrics, training_df = _find_first_nonempty(
            model_dir, "training_metrics*.log", _parse_metrics_log
        )

        if not train_output_df.empty:
            train_output_agg = _aggregate_by_ts(train_output_df, TRAIN_OUTPUT_METRICS)
            train_output_metrics = _summarize_metrics(
                train_output_agg,
                TRAIN_OUTPUT_METRICS,
                ts_ref,
                k_eval,
                best_metric="ep_rew_mean",
            )
            rows.append({
                "model_id": model_id,
                "eval_mode": "train_output",
                "ts_first": ts_ref[0] if ts_ref else None,
                "ts_last": ts_ref[-1] if ts_ref else None,
                "ts_step": ts_step,
                "metrics": train_output_metrics,
                **exp_params,
            })

        if not training_df.empty:
            training_agg = _aggregate_by_ts(training_df, TRAINING_METRICS)
            training_metrics_summary = _summarize_metrics(
                training_agg,
                TRAINING_METRICS,
                ts_ref,
                k_eval,
                allow_floor_ts=True,
            )
            rows.append({
                "model_id": model_id,
                "eval_mode": "train_metrics",
                "ts_first": ts_ref[0] if ts_ref else None,
                "ts_last": ts_ref[-1] if ts_ref else None,
                "ts_step": ts_step,
                "metrics": training_metrics_summary,
                **exp_params,
            })

        if not training_only:
            eval_logs = _collect_eval_logs(model_dir)
            for mode, log_path in eval_logs.items():
                eval_df = _parse_metrics_log(log_path)
                if eval_df.empty:
                    continue
                eval_agg = _aggregate_by_ts(eval_df, EVAL_METRICS)
                eval_metrics_summary = _summarize_metrics(eval_agg, EVAL_METRICS, ts_ref, k_eval)
                rows.append({
                    "model_id": model_id,
                    "eval_mode": mode,
                    "ts_first": ts_ref[0] if ts_ref else None,
                    "ts_last": ts_ref[-1] if ts_ref else None,
                    "ts_step": ts_step,
                    "metrics": eval_metrics_summary,
                    **exp_params,
                })

    _write_summary_log(output_file, rows, ts_ref, ts_step)

    csv_rows = []
    for row in rows:
        for metric, stats in row["metrics"].items():
            csv_rows.append({
                "model_id": row["model_id"],
                "eval_mode": row["eval_mode"],
                "metric": metric,
                "ts_best": stats.ts_best,
                "ts_last": stats.ts_last,
                "best": stats.best,
                "best_std": stats.best_std,
                "last": stats.last,
                "last_std": stats.last_std,
                "last_k_mean": stats.last_k_mean,
                "last_k_std": stats.last_k_std,
                "auc": stats.auc,
                "ts_first": row["ts_first"],
                "ts_last_ref": row["ts_last"],
                "ts_step": row["ts_step"],
                "use_xy_pickup": row["use_xy_pickup"],
                "normalize_features": row["normalize_features"],
                "logit_temperature": row["logit_temperature"],
                "noop_init": row["noop_init"],
                "freeze_noop_logit": row["freeze_noop_logit"],
            })

    pd.DataFrame(csv_rows).to_csv(output_csv, index=False)
    print(f"[OK] Wrote log: {output_file}")
    print(f"[OK] Wrote CSV: {output_csv}")


if __name__ == "__main__":
    main()
