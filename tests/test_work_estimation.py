import tempfile
import unittest
from pathlib import Path

from aggregate_metrics_logs import MetricStats, ParsedLog
from estimate_work_metrics import (
    WORK_MODEL_NAME,
    canonical_policy_name,
    canonical_route_construction_name,
    enrich_rows,
    insertion_pairs,
    proposer_work,
    resolver_work,
)


class WorkEstimationTests(unittest.TestCase):
    def test_parse_instance_metadata_to_robots_and_capacity(self):
        parsed = ParsedLog(
            path=Path("metrics_v_cap2_wave.log"),
            metadata={"instance": "wave_demand_cap2_taxis6.rou.xml"},
            per_seed_rows=[],
            summary_by_policy={"random": {"rew": MetricStats(mean=1.0, std=0.0)}},
        )
        self.assertEqual(parsed.num_robots, 6)
        self.assertEqual(parsed.max_robot_capacity, 2)

    def test_simple_policy_and_capacity_resolver(self):
        candidate_scan = 100.0
        insertion = insertion_pairs(4.0)
        p_work = proposer_work(
            policy_name="pickup_distance",
            candidate_scan_work=candidate_scan,
            insertion_pairs_value=insertion,
            competition_factor=1.0,
            joint_multiplier=1.5,
        )
        r_work = resolver_work(
            resolver_name="capacity",
            active_proposal_work=40.0,
            insertion_pairs_value=insertion,
            joint_multiplier=1.5,
            decisions_per_episode=10.0,
            num_robots=6.0,
            mean_candidates=3.0,
        )
        self.assertEqual(p_work, 100.0)
        self.assertEqual(r_work, 40.0)

    def test_predicted_reward_joint_with_predicted_resolver(self):
        candidate_scan = 20.0
        insertion = insertion_pairs(4.0)
        p_work = proposer_work(
            policy_name="predicted_reward_joint",
            candidate_scan_work=candidate_scan,
            insertion_pairs_value=insertion,
            competition_factor=1.0,
            joint_multiplier=1.5,
            route_construction_name="reward_aligned",
        )
        r_work = resolver_work(
            resolver_name="predicted_reward",
            active_proposal_work=12.0,
            insertion_pairs_value=insertion,
            joint_multiplier=1.5,
            decisions_per_episode=8.0,
            num_robots=6.0,
            mean_candidates=2.0,
            route_construction_name="reward_aligned",
        )
        self.assertAlmostEqual(p_work, candidate_scan * insertion * 1.5)
        self.assertAlmostEqual(r_work, 12.0 * insertion)

    def test_route_construction_changes_insertion_multiplier(self):
        candidate_scan = 20.0
        insertion = insertion_pairs(4.0)

        nearest_policy = proposer_work(
            policy_name="predicted_reward",
            candidate_scan_work=candidate_scan,
            insertion_pairs_value=insertion,
            competition_factor=1.0,
            joint_multiplier=1.5,
            route_construction_name="nearest",
        )
        reward_aligned_policy = proposer_work(
            policy_name="predicted_reward",
            candidate_scan_work=candidate_scan,
            insertion_pairs_value=insertion,
            competition_factor=1.0,
            joint_multiplier=1.5,
            route_construction_name="reward_aligned",
        )

        nearest_resolver = resolver_work(
            resolver_name="predicted_reward",
            active_proposal_work=12.0,
            insertion_pairs_value=insertion,
            joint_multiplier=1.5,
            decisions_per_episode=8.0,
            num_robots=6.0,
            mean_candidates=2.0,
            route_construction_name="nearest",
        )
        reward_aligned_resolver = resolver_work(
            resolver_name="predicted_reward",
            active_proposal_work=12.0,
            insertion_pairs_value=insertion,
            joint_multiplier=1.5,
            decisions_per_episode=8.0,
            num_robots=6.0,
            mean_candidates=2.0,
            route_construction_name="reward_aligned",
        )

        self.assertEqual(canonical_route_construction_name("deadline_travel"), "reward_aligned")
        self.assertAlmostEqual(nearest_policy, candidate_scan)
        self.assertAlmostEqual(reward_aligned_policy, candidate_scan * insertion)
        self.assertAlmostEqual(nearest_resolver, 12.0)
        self.assertAlmostEqual(reward_aligned_resolver, 12.0 * insertion)

    def test_hungarian_scales_with_decisions(self):
        low = resolver_work(
            resolver_name="hungarian",
            active_proposal_work=10.0,
            insertion_pairs_value=15.0,
            joint_multiplier=1.5,
            decisions_per_episode=5.0,
            num_robots=6.0,
            mean_candidates=2.0,
        )
        high = resolver_work(
            resolver_name="hungarian",
            active_proposal_work=10.0,
            insertion_pairs_value=15.0,
            joint_multiplier=1.5,
            decisions_per_episode=10.0,
            num_robots=6.0,
            mean_candidates=2.0,
        )
        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        self.assertGreater(high, low)
        self.assertAlmostEqual(high, 2.0 * low)

    def test_competition_prefers_cemn_proxy(self):
        rows = [
            {
                "source_file": "x.log",
                "scenario": "wave",
                "resolver": "capacity",
                "pol": "proposal_joint_competition",
                "route_construction": "nearest",
                "num_robots": "6",
                "max_robot_capacity": "2",
                "mcand": "2",
                "msd": "3",
                "dstep": "3",
                "noop": "0",
                "ovrlap": "0.9",
                "shared": "0.9",
                "cemn": "0.1",
            }
        ]
        enriched, enriched_count, skipped_count = enrich_rows(
            rows,
            joint_multiplier=1.5,
            route_stops_override=None,
            default_num_robots=None,
            unknown_mode="warn",
        )
        self.assertEqual(enriched_count, 1)
        self.assertEqual(skipped_count, 0)
        out = enriched[0]
        self.assertEqual(out["work_model"], WORK_MODEL_NAME)
        # Uses max(cemn, ovrlap*shared), so 1 + max(0.1, 0.81) = 1.81
        self.assertAlmostEqual(float(out["work_competition_factor"]), 1.81, places=6)

    def test_unknown_policy_warn_mode_blanks_work_fields(self):
        rows = [
            {
                "source_file": "x.log",
                "scenario": "wave",
                "resolver": "capacity",
                "pol": "new_unknown_policy",
                "route_construction": "nearest",
                "num_robots": "6",
                "max_robot_capacity": "2",
                "mcand": "2",
                "msd": "3",
                "dstep": "3",
                "noop": "0",
            }
        ]
        enriched, enriched_count, skipped_count = enrich_rows(
            rows,
            joint_multiplier=1.5,
            route_stops_override=None,
            default_num_robots=None,
            unknown_mode="warn",
        )
        self.assertEqual(enriched_count, 0)
        self.assertEqual(skipped_count, 1)
        out = enriched[0]
        self.assertEqual(out["work_total"], "")
        self.assertIn("unknown_policy", out["work_warning"])

    def test_alias_normalization(self):
        self.assertEqual(canonical_policy_name("predicted-reward-joint"), "predicted_reward_joint")


class WorkPlotSmokeTests(unittest.TestCase):
    def test_detect_metrics_excludes_work_columns(self):
        try:
            from plot_metrics_wide import detect_metrics
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed in this environment")

        fieldnames = [
            "source_file",
            "scenario",
            "resolver",
            "pol",
            "rew",
            "rew_std",
            "work_total",
            "work_resolver",
            "work_model",
            "num_robots",
            "max_robot_capacity",
        ]
        metrics = detect_metrics(fieldnames)
        self.assertEqual(metrics, ["rew"])

    def test_work_cmp_writes_rew_plot(self):
        try:
            from plot_metrics_wide import plot_work_cmp
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed in this environment")

        rows = [
            {
                "scenario": "corridor_asymmetric",
                "protocol": "admission",
                "route_construction": "nearest",
                "resolver": "capacity",
                "pol": "pickup_deadline",
                "rew": 10.0,
                "rew_std": 1.0,
                "work_total": 100.0,
            },
            {
                "scenario": "corridor_asymmetric",
                "protocol": "admission",
                "route_construction": "nearest",
                "resolver": "hungarian",
                "pol": "pickup_deadline",
                "rew": 11.0,
                "rew_std": 0.8,
                "work_total": 200.0,
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            saved = plot_work_cmp(
                rows,
                ["rew"],
                output_dir,
                work_x="relative",
                work_x_padding=0.06,
                pareto=True,
                annotate_work=False,
            )
            self.assertTrue(any(path.name == "rew_work_cmp.png" for path in saved))

    def test_protocol_cmp_grouped_writes_rew_plot(self):
        try:
            from plot_metrics_wide import plot_protocol_cmp_grouped
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed in this environment")

        rows = [
            {
                "scenario": "wave",
                "resolver": "predicted_reward",
                "protocol": "forced",
                "pol": "predicted_reward",
                "rew": 10.0,
                "rew_std": 1.0,
            },
            {
                "scenario": "wave",
                "resolver": "predicted_reward",
                "protocol": "admission",
                "pol": "predicted_reward",
                "rew": 11.0,
                "rew_std": 1.0,
            },
            {
                "scenario": "wave",
                "resolver": "predicted_reward",
                "protocol": "forced",
                "pol": "predicted_reward_joint",
                "rew": 12.0,
                "rew_std": 1.1,
            },
            {
                "scenario": "wave",
                "resolver": "predicted_reward",
                "protocol": "admission",
                "pol": "predicted_reward_joint",
                "rew": 13.0,
                "rew_std": 1.2,
            },
            {
                "scenario": "wave",
                "resolver": "predicted_reward",
                "protocol": "forced",
                "pol": "proposal_joint_competition",
                "rew": 14.0,
                "rew_std": 0.8,
            },
            {
                "scenario": "wave",
                "resolver": "predicted_reward",
                "protocol": "admission",
                "pol": "proposal_joint_competition",
                "rew": 15.0,
                "rew_std": 0.9,
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            saved = plot_protocol_cmp_grouped(rows, ["rew"], output_dir)
            self.assertTrue(any(path.name == "rew_protocol_cmp_grouped.png" for path in saved))
            self.assertTrue((output_dir / "protocol_cmp").exists())

    def test_protocol_cmp_grouped_resolver_alias_match(self):
        try:
            from plot_metrics_wide import plot_protocol_cmp_grouped
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed in this environment")

        rows = [
            {
                "scenario": "randdest",
                "resolver": "closest",
                "protocol": "forced",
                "pol": "predicted_reward",
                "rew": 10.0,
                "rew_std": 0.5,
            },
            {
                "scenario": "randdest",
                "resolver": "closest_then_capacity",
                "protocol": "admission",
                "pol": "predicted_reward",
                "rew": 11.0,
                "rew_std": 0.6,
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            saved = plot_protocol_cmp_grouped(rows, ["rew"], output_dir)
            self.assertTrue(any(path.name == "rew_protocol_cmp_grouped.png" for path in saved))

    def test_tight_log_x_limits_helper(self):
        try:
            import matplotlib.pyplot as plt
            from plot_metrics_wide import set_tight_log_x_limits
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed in this environment")

        x_values = [1.0, 1.2, 6.0, 10.0, 42.0]
        figure, axis = plt.subplots(figsize=(4, 3))
        axis.set_xscale("log")
        axis.scatter(x_values, [1.0] * len(x_values))
        set_tight_log_x_limits(axis, x_values, padding_fraction=0.06)

        lower, upper = axis.get_xlim()
        self.assertGreater(lower, 0.5)
        self.assertGreater(upper, 42.0)
        self.assertLess(upper, 100.0)
        self.assertEqual(axis.get_xscale(), "log")
        self.assertTrue(all(lower <= value <= upper for value in x_values))

        plt.close(figure)

    def test_exclude_policies_normalization_filters_space_and_comma_inputs(self):
        try:
            from plot_metrics_wide import _filtered_rows, _normalize_policy_set
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed in this environment")

        rows = [
            {"resolver": "capacity", "pol": "pickup_deadline", "rew": 1.0, "work_total": 10.0},
            {"resolver": "capacity", "pol": "pickup_deadline_distance", "rew": 2.0, "work_total": 11.0},
            {"resolver": "capacity", "pol": "unique", "rew": 3.0, "work_total": 12.0},
            {"resolver": "capacity", "pol": "proposal_joint_competition", "rew": 4.0, "work_total": 13.0},
            {"resolver": "capacity", "pol": "predicted_reward", "rew": 5.0, "work_total": 14.0},
        ]

        # Mimics CLI usage where several values are passed as one comma-separated token.
        exclude_policies = _normalize_policy_set([
            "pickup deadline, pickup deadline distance, unique, proposal joint competition"
        ])
        filtered = _filtered_rows(rows, exclude_resolvers=set(), exclude_policies=exclude_policies)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["pol"], "predicted_reward")


if __name__ == "__main__":
    unittest.main()
