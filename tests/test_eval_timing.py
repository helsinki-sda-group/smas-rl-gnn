import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.eval_timing import STEP_COLUMNS, TimingConfig, TimingRunCollector


class EvalTimingTests(unittest.TestCase):
    def test_step_columns_stable_subset(self):
        required = {
            "method",
            "policy",
            "proposer",
            "resolver",
            "protocol",
            "inference_mode",
            "seed",
            "episode",
            "decision_idx",
            "proposal_ns",
            "resolution_ns",
            "simulation_ns",
            "decision_total_ns",
            "other_ns",
            "gnn_obs_build_ns",
            "gnn_tensor_prepare_ns",
            "gnn_graph_prepare_ns",
            "gnn_policy_total_ns",
            "gnn_actor_ns",
            "gnn_critic_ns",
            "gnn_action_ns",
            "gnn_action_mapping_ns",
            "n_actor_graphs",
            "n_actor_nonempty_graphs",
            "n_actor_nodes",
            "n_actor_edges",
            "n_actor_candidates",
            "gnn_actor_amortized_robot_ns",
            "gnn_actor_path_accounted_amortized_robot_ns",
            "gnn_actor_only_proposal_est_amortized_robot_ns",
        }
        self.assertTrue(required.issubset(set(STEP_COLUMNS)))

    def test_derived_per_robot_fields_and_zero_division(self):
        collector = TimingRunCollector(
            config=TimingConfig(enabled=True, internal_gnn=True, warmup_episodes=0),
            static_meta={
                "scenario": "cfg",
                "method": "gnn",
                "policy": "p",
                "proposer": "gnn",
                "resolver": "capacity",
                "protocol": "forced",
                "inference_mode": "actor_critic",
                "seed": 1,
                "device": "cpu",
            },
        )

        collector.add_step(
            {
                "method": "gnn",
                "policy": "p",
                "proposer": "gnn",
                "resolver": "capacity",
                "protocol": "forced",
                "inference_mode": "actor_critic",
                "seed": 1,
                "episode": 0,
                "decision_idx": 0,
                "proposal_ns": 1000,
                "resolution_ns": 200,
                "simulation_ns": 300,
                "decision_total_ns": 1700,
                "gnn_obs_build_ns": 100,
                "gnn_tensor_prepare_ns": 100,
                "gnn_graph_prepare_ns": 100,
                "gnn_actor_ns": 300,
                "gnn_critic_ns": 200,
                "gnn_action_ns": 50,
                "gnn_action_mapping_ns": 50,
                "n_actor_graphs": 5,
                "n_actor_nonempty_graphs": 3,
                "n_actor_nodes": 20,
                "n_actor_edges": 40,
                "n_actor_candidates": 8,
            }
        )

        row = collector.rows[-1]
        self.assertEqual(row["gnn_actor_amortized_robot_ns"], 60)
        # (100 + 100 + 100 + 300 + 50 + 50) / 5 = 140
        self.assertEqual(row["gnn_actor_path_accounted_amortized_robot_ns"], 140)
        # max(0, 1000 - 200) / 5 = 160
        self.assertEqual(row["gnn_actor_only_proposal_est_amortized_robot_ns"], 160)

        collector.add_step(
            {
                "method": "gnn",
                "policy": "p",
                "proposer": "gnn",
                "resolver": "capacity",
                "protocol": "forced",
                "inference_mode": "actor_critic",
                "seed": 1,
                "episode": 0,
                "decision_idx": 1,
                "proposal_ns": 10,
                "resolution_ns": 10,
                "simulation_ns": 10,
                "decision_total_ns": 30,
                "n_actor_graphs": 0,
            }
        )
        row_zero = collector.rows[-1]
        self.assertEqual(row_zero["gnn_actor_amortized_robot_ns"], 0)
        self.assertEqual(row_zero["gnn_actor_path_accounted_amortized_robot_ns"], 0)
        self.assertEqual(row_zero["gnn_actor_only_proposal_est_amortized_robot_ns"], 0)

    def test_summary_excludes_warmup_from_primary_aggregates(self):
        collector = TimingRunCollector(
            config=TimingConfig(enabled=True, internal_gnn=True, warmup_episodes=1),
            static_meta={
                "scenario": "cfg",
                "python_version": "py",
                "torch_version": "",
                "torch_geometric_version": "",
                "stable_baselines3_version": "",
                "torch_num_threads": 1,
                "omp_num_threads": "",
                "mkl_num_threads": "",
                "host_name": "h",
                "cpu_model": "c",
                "git_commit": "",
                "internal_gnn_timing": 1,
                "actor_edge_count_convention": "pre_encoder_no_self_loops",
                "actor_graphs_include_empty_candidates": 1,
            },
        )

        base = {
            "method": "gnn",
            "policy": "p",
            "proposer": "gnn",
            "resolver": "capacity",
            "protocol": "forced",
            "inference_mode": "actor_critic",
            "seed": 1,
            "episode": 0,
            "device": "cpu",
            "n_actor_graphs": 2,
        }

        collector.add_step({**base, "decision_idx": 0, "warmup": 1, "proposal_ns": 100, "resolution_ns": 10, "simulation_ns": 10, "decision_total_ns": 120, "gnn_actor_ns": 20})
        collector.add_step({**base, "decision_idx": 1, "warmup": 0, "proposal_ns": 200, "resolution_ns": 20, "simulation_ns": 20, "decision_total_ns": 240, "gnn_actor_ns": 40})

        with tempfile.TemporaryDirectory() as td:
            summary_path = collector.write_summary_csv(td)
            df = pd.read_csv(summary_path)
            self.assertEqual(int(df.loc[0, "warmup_decisions"]), 1)
            self.assertEqual(int(df.loc[0, "measured_decisions"]), 1)
            self.assertAlmostEqual(float(df.loc[0, "proposal_total_ms"]), 0.0002, places=7)


if __name__ == "__main__":
    unittest.main()
