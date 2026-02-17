import argparse
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
from stable_baselines3 import PPO

from rt_gnn_rl.policy.sb3_gnn_policy import RTGNNPolicy
from sumo_rl_rs.environment.ridepool_rt_env import RidepoolRTEnv
from sumo_rl_rs.environment.rl_controller_adapter import RLControllerAdapter
from sumo_rl_rs.logging.ridepool_logger import RidepoolLogger, RidepoolLogConfig
from utils.feature_fns import make_feature_fn
from utils.metrics_calculator import (
    append_metrics_log,
    append_metrics_summary,
    compute_episode_metrics_from_logs,
    compute_metrics_summary,
    ensure_metrics_log,
)
from utils.sumo_bootstrap import _build_args, start_sumo


def _parse_seeds(seeds_arg: str) -> List[int]:
    if not seeds_arg:
        return [
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
            1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
        ]
    return [int(s.strip()) for s in seeds_arg.split(",") if s.strip()]


def _list_model_paths(model_path: Path) -> List[Path]:
    if model_path.is_file():
        return [model_path]
    if model_path.is_dir():
        return sorted(model_path.rglob("model.zip"))
    raise FileNotFoundError(f"Model path not found: {model_path}")


def _run_single_eval_episode(
    model: PPO,
    sumo_cfg: str,
    seed: int,
    rp_logger: RidepoolLogger,
    eval_ep_idx: int,
    num_robots: int,
    k_max: int,
    vicinity_m: float,
    max_steps: int,
    min_episode_steps: int,
    max_wait_delay_s: float,
    max_travel_delay_s: float,
    max_robot_capacity: int,
    decision_dt: int,
    port: int,
    use_gui: bool,
    deterministic: bool,
    sorted_candidates: bool,
) -> str:
    eval_conn = start_sumo(
        sumo_cfg,
        use_gui=use_gui,
        extra_args=["--seed", str(seed), "--device.taxi.dispatch-algorithm", "traci"],
        label=f"holdout_{seed}",
        port=port,
    )

    try:
        args = _build_args(sumo_cfg, [
            "--seed",
            str(seed),
            "--device.taxi.dispatch-algorithm",
            "traci",
        ])

        def reset_fn() -> None:
            eval_conn.load(args)

        controller = RLControllerAdapter(
            sumo=eval_conn,
            reset_fn=reset_fn,
            k_max=k_max,
            vicinity_m=vicinity_m,
            sorted_candidates=sorted_candidates,
            completion_mode="dropoff",
            max_steps=max_steps,
            min_episode_steps=min_episode_steps,
            serve_to_empty=True,
            require_seen_reservation=True,
            max_wait_delay_s=max_wait_delay_s,
            max_travel_delay_s=max_travel_delay_s,
            max_robot_capacity=max_robot_capacity,
            logger=rp_logger,
        )
        controller._ep_idx = eval_ep_idx - 1

        feature_fn = make_feature_fn(controller)
        env = RidepoolRTEnv(
            controller,
            R=num_robots,
            K_max=k_max,
            N_max=16,
            E_max=64,
            F=11,
            G=0,
            feature_fn=feature_fn,
            global_stats_fn=None,
            decision_dt=decision_dt,
        )

        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

        # Ensure logger flushes all CSV files
        try:
            rp_logger.finalize()
        except Exception:
            pass

        episode_dir = getattr(rp_logger, "last_ep_dir", None) or rp_logger.ep_dir

        try:
            env.close()
        except Exception:
            pass

    finally:
        try:
            eval_conn.close()
        except Exception:
            pass

    return episode_dir


def _write_holdout_summary(
    summary_path: Path,
    model_tag: str,
    num_seeds: int,
    mean_metrics,
    std_metrics,
) -> None:
    if not summary_path.exists():
        summary_path.write_text(
            "model | seeds | rew_mean | rew_std | comp_rate | comp_rate_std | pickup_rate | pickup_rate_std | obsolete_rate | pickup_violated_rate | noop_fraction | macro_reward_mean\n",
            encoding="utf-8",
        )

    line = (
        f"{model_tag} | {num_seeds:>5} | "
        f"{mean_metrics.reward_sum:>8.2f} | {std_metrics.reward_sum:>7.2f} | "
        f"{mean_metrics.completion_rate:>9.2f} | {std_metrics.completion_rate:>13.2f} | "
        f"{mean_metrics.pickup_rate:>11.2f} | {std_metrics.pickup_rate:>15.2f} | "
        f"{mean_metrics.obsolete_rate:>13.2f} | {mean_metrics.pickup_violated_rate:>20.2f} | "
        f"{mean_metrics.noop_fraction:>13.2f} | {mean_metrics.macro_reward_mean:>17.2f}\n"
    )
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(line)


def _eval_model(
    model_path: Path,
    out_base_dir: Path,
    seeds: Iterable[int],
    sumo_cfg: str,
    use_gui: bool,
    num_robots: int,
    k_max: int,
    vicinity_m: float,
    max_steps: int,
    min_episode_steps: int,
    max_wait_delay_s: float,
    max_travel_delay_s: float,
    max_robot_capacity: int,
    decision_dt: int,
    port_base: int,
    deterministic: bool,
    device: str,
    sorted_candidates: bool,
) -> None:
    run_name = model_path.parent.name if model_path.name == "model.zip" else model_path.stem
    run_dir = out_base_dir / f"holdout_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(str(model_path), device=device)
    model.policy.eval()  # Set policy to eval mode (disables dropout, batchnorm, etc.)

    rp_logger = RidepoolLogger(
        RidepoolLogConfig(
            out_dir=str(out_base_dir),
            run_name=run_dir.name,
            erase_run_dir_on_start=False,
            erase_episode_dir_on_start=True,
            console_debug=False,
        )
    )

    metrics_log_path = run_dir / "training_metrics.log"
    ensure_metrics_log(str(metrics_log_path), overwrite=True)

    episode_metrics = []
    for idx, seed in enumerate(seeds):
        episode_dir = _run_single_eval_episode(
            model=model,
            sumo_cfg=sumo_cfg,
            seed=seed,
            rp_logger=rp_logger,
            eval_ep_idx=idx,
            num_robots=num_robots,
            k_max=k_max,
            vicinity_m=vicinity_m,
            max_steps=max_steps,
            min_episode_steps=min_episode_steps,
            max_wait_delay_s=max_wait_delay_s,
            max_travel_delay_s=max_travel_delay_s,
            max_robot_capacity=max_robot_capacity,
            decision_dt=decision_dt,
            port=port_base + idx,
            use_gui=use_gui,
            deterministic=deterministic,
            sorted_candidates=sorted_candidates,
        )

        metrics = compute_episode_metrics_from_logs(
            episode_dir=episode_dir,
            episode_info={},
            policy=str(idx),
            seed=seed,
            num_robots=num_robots,
        )
        append_metrics_log(str(metrics_log_path), metrics)
        episode_metrics.append(metrics)

    append_metrics_summary(str(metrics_log_path), episode_metrics)

    mean_metrics, std_metrics = compute_metrics_summary(episode_metrics)
    summary_path = out_base_dir / "holdout_metrics.log"
    _write_holdout_summary(summary_path, run_dir.name, len(list(seeds)), mean_metrics, std_metrics)

    try:
        rp_logger.close()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model.zip snapshots on holdout seeds.")
    parser.add_argument("--model", required=True, help="Path to model.zip or directory containing snapshots.")
    parser.add_argument("--out-dir", default="runs/holdout_eval", help="Base output directory.")
    parser.add_argument("--seeds", default="", help="Comma-separated seeds list.")
    parser.add_argument("--sumo-cfg", default="configs/small_net.sumocfg")
    parser.add_argument("--use-gui", action="store_true")
    parser.add_argument("--r", type=int, default=5)
    parser.add_argument("--k-max", type=int, default=3)
    parser.add_argument("--vicinity-m", type=float, default=2000.0)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--min-episode-steps", type=int, default=100)
    parser.add_argument("--max-wait-delay-s", type=float, default=240.0)
    parser.add_argument("--max-travel-delay-s", type=float, default=900.0)
    parser.add_argument("--max-robot-capacity", type=int, default=2)
    parser.add_argument("--decision-dt", type=int, default=60)
    parser.add_argument("--port-base", type=int, default=8816)
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
    parser.add_argument("--sorted", action="store_true", help="Sort candidates by pickup distance (default: randomized)")
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    model_path = Path(args.model).expanduser()
    out_base_dir = Path(args.out_dir)
    out_base_dir.mkdir(parents=True, exist_ok=True)

    model_paths = _list_model_paths(model_path)
    if not model_paths:
        raise FileNotFoundError(f"No model.zip files found in: {model_path}")

    for mp in model_paths:
        _eval_model(
            model_path=mp,
            out_base_dir=out_base_dir,
            seeds=seeds,
            sumo_cfg=args.sumo_cfg,
            use_gui=args.use_gui,
            num_robots=args.r,
            k_max=args.k_max,
            vicinity_m=args.vicinity_m,
            max_steps=args.max_steps,
            min_episode_steps=args.min_episode_steps,
            max_wait_delay_s=args.max_wait_delay_s,
            max_travel_delay_s=args.max_travel_delay_s,
            max_robot_capacity=args.max_robot_capacity,
            decision_dt=args.decision_dt,
            port_base=args.port_base,
            deterministic=args.deterministic,
            sorted_candidates=args.sorted,
            device=args.device,
        )


if __name__ == "__main__":
    main()
