import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PlotMetricsWideTimeUnitTests(unittest.TestCase):
    def test_alias_canonicalization(self):
        from plot_metrics_wide import (
            _canonical_policy_join_name,
            _canonical_resolver_name,
            _protocol_alias_from_text,
            _route_alias_from_text,
        )

        self.assertEqual(_canonical_resolver_name("closest"), "closest_then_capacity")
        self.assertEqual(_canonical_resolver_name("ctc"), "closest_then_capacity")
        self.assertEqual(_canonical_resolver_name("closest_than_capacity"), "closest_then_capacity")
        self.assertEqual(_protocol_alias_from_text("admission_aware"), "aa")
        self.assertEqual(_protocol_alias_from_text("admission"), "aa")
        self.assertEqual(_protocol_alias_from_text("f"), "forced")
        self.assertEqual(_route_alias_from_text("nearest"), "nr")
        self.assertEqual(_route_alias_from_text("reward-aligned"), "ra")
        self.assertEqual(_canonical_policy_join_name("proposal_joint_competition"), "proposal_joint_completion")

    def test_filename_inference_randdest_nr_aa(self):
        from plot_metrics_wide import (
            _infer_protocol_alias_from_path,
            _infer_route_alias_from_path,
            _infer_scenario_from_path,
        )

        text = "timing_summary_v2000_ms3000_mwd300_mtd240_cap2_randdest_closest_nr_aa.csv"
        scenario = _infer_scenario_from_path(text, {"randdest", "wave"})
        route = _infer_route_alias_from_path(text)
        protocol = _infer_protocol_alias_from_path(text)

        self.assertEqual(scenario, "randdest")
        self.assertEqual(route, "nr")
        self.assertEqual(protocol, "aa")

    def _canonical_row(self, *, measured, proposal_total, proposal_mean, resolution_total, resolution_mean, seed, episode, proposer="predicted_reward", resolver="closest_then_capacity", method="baseline", inference_mode="na", source="a.csv", phases=None):
        phases = phases or {}
        row = {
            "scenario_canonical": "randdest",
            "route_canonical": "nr",
            "protocol_canonical": "aa",
            "proposer_canonical": proposer,
            "resolver_canonical": resolver,
            "method_canonical": method,
            "inference_mode_canonical": inference_mode,
            "seed_canonical": str(seed),
            "episode_canonical": str(episode),
            "timing_protocol_canonical": "shared_outer_v1",
            "source_timing_file": source,
            "measured_decisions": float(measured),
            "proposal_total_ms": float(proposal_total),
            "proposal_mean_ms": float(proposal_mean),
            "resolution_total_ms": float(resolution_total),
            "resolution_mean_ms": float(resolution_mean),
            "timing_protocol": "shared_outer_v1",
            "device": "cpu",
            "torch_num_threads": "1",
            "omp_num_threads": "1",
            "mkl_num_threads": "1",
            "cpu_model": "x",
            "host_name": "h",
        }
        for phase in [
            "env_pre_controller",
            "pre_step_sync",
            "proposal",
            "resolution",
            "commit_dispatch",
            "simulation",
            "post_step_logging",
            "other",
            "decision",
        ]:
            total_key = f"{phase}_total_ms"
            mean_key = f"{phase}_mean_ms"
            if total_key in phases:
                row[total_key] = phases[total_key]
            if mean_key in phases:
                row[mean_key] = phases[mean_key]
        return row

    def test_pooled_means_with_unequal_decisions(self):
        from plot_metrics_wide import _aggregate_timing_groups

        rows = [
            self._canonical_row(measured=10, proposal_total=100, proposal_mean=10, resolution_total=50, resolution_mean=5, seed=1, episode=1),
            self._canonical_row(measured=20, proposal_total=300, proposal_mean=15, resolution_total=100, resolution_mean=5, seed=2, episode=1),
        ]
        groups = _aggregate_timing_groups(rows)
        self.assertEqual(len(groups), 1)
        g = groups[0]
        self.assertAlmostEqual(g["proposal_time_ms"], (100 + 300) / (10 + 20), places=8)
        self.assertAlmostEqual(g["resolution_time_ms"], (50 + 100) / (10 + 20), places=8)
        self.assertAlmostEqual(g["allocation_time_ms"], (100 + 300 + 50 + 100) / (10 + 20), places=8)

    def test_across_run_std(self):
        from plot_metrics_wide import _aggregate_timing_groups

        rows = [
            self._canonical_row(measured=10, proposal_total=100, proposal_mean=10, resolution_total=50, resolution_mean=5, seed=1, episode=1),
            self._canonical_row(measured=20, proposal_total=300, proposal_mean=15, resolution_total=100, resolution_mean=9, seed=2, episode=1),
        ]
        g = _aggregate_timing_groups(rows)[0]
        self.assertGreater(g["proposal_time_std_ms"], 0.0)
        self.assertGreater(g["resolution_time_std_ms"], 0.0)
        self.assertGreater(g["allocation_time_std_ms"], 0.0)

    def test_join_proposer_to_metrics_pol(self):
        from plot_metrics_wide import _join_metrics_with_timing

        metrics_rows = [
            {
                "scenario": "randdest",
                "route_construction": "nearest",
                "protocol": "admission",
                "resolver": "closest_then_capacity",
                "pol": "predicted_reward",
                "rew": 10.0,
                "rew_std": 0.5,
            }
        ]
        timing_groups = [
            {
                "scenario": "randdest",
                "route": "nr",
                "protocol": "aa",
                "proposer": "predicted_reward",
                "resolver": "closest_then_capacity",
                "method": "baseline",
                "inference_mode": "na",
                "allocation_time_ms": 1.0,
            }
        ]
        joined, missing, unmatched = _join_metrics_with_timing(metrics_rows, timing_groups)
        self.assertEqual(len(joined), 1)
        self.assertFalse(missing)
        self.assertFalse(unmatched)

    def test_resolver_alias_matching(self):
        from plot_metrics_wide import _join_metrics_with_timing

        metrics_rows = [
            {
                "scenario": "randdest",
                "route_construction": "nearest",
                "protocol": "admission",
                "resolver": "closest",
                "pol": "predicted_reward",
                "rew": 10.0,
                "rew_std": 0.5,
            }
        ]
        timing_groups = [
            {
                "scenario": "randdest",
                "route": "nr",
                "protocol": "aa",
                "proposer": "predicted_reward",
                "resolver": "closest_then_capacity",
                "method": "baseline",
                "inference_mode": "na",
                "allocation_time_ms": 1.0,
            }
        ]
        joined, missing, unmatched = _join_metrics_with_timing(metrics_rows, timing_groups)
        self.assertEqual(len(joined), 1)
        self.assertFalse(missing)
        self.assertFalse(unmatched)

    def test_duplicate_detection(self):
        from plot_metrics_wide import _detect_duplicate_timing_rows

        row1 = self._canonical_row(measured=10, proposal_total=100, proposal_mean=10, resolution_total=50, resolution_mean=5, seed=1, episode=1, source="a.csv")
        row2 = self._canonical_row(measured=10, proposal_total=100, proposal_mean=10, resolution_total=50, resolution_mean=5, seed=1, episode=1, source="b.csv")
        with self.assertRaises(ValueError):
            _detect_duplicate_timing_rows([row1, row2])

    def test_missing_matches_are_skipped_not_zero_filled(self):
        from plot_metrics_wide import _join_metrics_with_timing

        metrics_rows = [
            {
                "scenario": "randdest",
                "route_construction": "nearest",
                "protocol": "admission",
                "resolver": "capacity",
                "pol": "predicted_reward",
                "rew": 10.0,
                "rew_std": 0.5,
            }
        ]
        timing_groups = []
        joined, missing, unmatched = _join_metrics_with_timing(metrics_rows, timing_groups)
        self.assertEqual(joined, [])
        self.assertEqual(len(missing), 1)
        self.assertEqual(len(unmatched), 0)

    def test_phase_pooled_mean_unequal_decisions(self):
        from plot_metrics_wide import _aggregate_timing_groups

        rows = [
            self._canonical_row(
                measured=10,
                proposal_total=100,
                proposal_mean=10,
                resolution_total=50,
                resolution_mean=5,
                seed=1,
                episode=1,
                phases={
                    "env_pre_controller_total_ms": 20.0,
                    "env_pre_controller_mean_ms": 2.0,
                    "pre_step_sync_total_ms": 10.0,
                    "pre_step_sync_mean_ms": 1.0,
                    "proposal_total_ms": 100.0,
                    "proposal_mean_ms": 10.0,
                    "resolution_total_ms": 50.0,
                    "resolution_mean_ms": 5.0,
                    "commit_dispatch_total_ms": 10.0,
                    "commit_dispatch_mean_ms": 1.0,
                    "simulation_total_ms": 200.0,
                    "simulation_mean_ms": 20.0,
                    "post_step_logging_total_ms": 5.0,
                    "post_step_logging_mean_ms": 0.5,
                    "other_total_ms": 5.0,
                    "other_mean_ms": 0.5,
                    "decision_total_ms": 400.0,
                    "decision_mean_ms": 40.0,
                },
            ),
            self._canonical_row(
                measured=20,
                proposal_total=300,
                proposal_mean=15,
                resolution_total=140,
                resolution_mean=7,
                seed=2,
                episode=1,
                phases={
                    "env_pre_controller_total_ms": 60.0,
                    "env_pre_controller_mean_ms": 3.0,
                    "pre_step_sync_total_ms": 40.0,
                    "pre_step_sync_mean_ms": 2.0,
                    "proposal_total_ms": 300.0,
                    "proposal_mean_ms": 15.0,
                    "resolution_total_ms": 140.0,
                    "resolution_mean_ms": 7.0,
                    "commit_dispatch_total_ms": 20.0,
                    "commit_dispatch_mean_ms": 1.0,
                    "simulation_total_ms": 500.0,
                    "simulation_mean_ms": 25.0,
                    "post_step_logging_total_ms": 10.0,
                    "post_step_logging_mean_ms": 0.5,
                    "other_total_ms": 30.0,
                    "other_mean_ms": 1.5,
                    "decision_total_ms": 1100.0,
                    "decision_mean_ms": 55.0,
                },
            ),
        ]
        g = _aggregate_timing_groups(rows)[0]
        self.assertAlmostEqual(g["env_pre_controller_mean_ms"], (20.0 + 60.0) / 30.0, places=8)

    def test_accounting_difference_and_decision_total_not_in_named_sum(self):
        from plot_metrics_wide import _aggregate_timing_groups

        row = self._canonical_row(
            measured=10,
            proposal_total=100,
            proposal_mean=10,
            resolution_total=50,
            resolution_mean=5,
            seed=1,
            episode=1,
            phases={
                "env_pre_controller_total_ms": 10.0,
                "env_pre_controller_mean_ms": 1.0,
                "pre_step_sync_total_ms": 10.0,
                "pre_step_sync_mean_ms": 1.0,
                "proposal_total_ms": 100.0,
                "proposal_mean_ms": 10.0,
                "resolution_total_ms": 50.0,
                "resolution_mean_ms": 5.0,
                "commit_dispatch_total_ms": 10.0,
                "commit_dispatch_mean_ms": 1.0,
                "simulation_total_ms": 200.0,
                "simulation_mean_ms": 20.0,
                "post_step_logging_total_ms": 10.0,
                "post_step_logging_mean_ms": 1.0,
                "other_total_ms": 10.0,
                "other_mean_ms": 1.0,
                "decision_total_ms": 700.0,
                "decision_mean_ms": 70.0,
            },
        )
        g = _aggregate_timing_groups([row])[0]
        named_sum = g["named_phase_sum_ms"]
        self.assertLess(named_sum, g["decision_total_mean_ms"])
        self.assertAlmostEqual(g["accounting_difference_ms"], g["decision_total_mean_ms"] - named_sum, places=8)

    def test_missing_optional_phase_is_graceful(self):
        from plot_metrics_wide import _aggregate_timing_groups

        row = self._canonical_row(
            measured=10,
            proposal_total=100,
            proposal_mean=10,
            resolution_total=50,
            resolution_mean=5,
            seed=1,
            episode=1,
            phases={
                "proposal_total_ms": 100.0,
                "proposal_mean_ms": 10.0,
                "resolution_total_ms": 50.0,
                "resolution_mean_ms": 5.0,
                "decision_total_ms": 200.0,
                "decision_mean_ms": 20.0,
            },
        )
        g = _aggregate_timing_groups([row])[0]
        self.assertIsNone(g["simulation_mean_ms"])

    def test_one_to_one_assignment_check(self):
        from plot_metrics_wide import _ensure_one_to_one_row_assignment

        row_a = self._canonical_row(measured=10, proposal_total=100, proposal_mean=10, resolution_total=50, resolution_mean=5, seed=1, episode=1)
        row_b = dict(row_a)
        row_b["resolver_canonical"] = "capacity"
        with self.assertRaises(ValueError):
            _ensure_one_to_one_row_assignment([row_a, row_b])

    def test_pooled_weighted_agreement_validation(self):
        from plot_metrics_wide import _proposal_group_means_or_error

        rows = [
            self._canonical_row(measured=10, proposal_total=100, proposal_mean=10, resolution_total=50, resolution_mean=5, seed=1, episode=1),
            self._canonical_row(measured=20, proposal_total=300, proposal_mean=15, resolution_total=100, resolution_mean=5, seed=2, episode=1),
        ]
        pooled, weighted, _, _, _, _ = _proposal_group_means_or_error(
            rows,
            group_key=("randdest", "nr", "aa", "baseline", "na", "predicted_reward", "closest_then_capacity"),
        )
        self.assertIsNotNone(pooled)
        self.assertIsNotNone(weighted)
        self.assertAlmostEqual(float(pooled), float(weighted), places=10)

    def test_workload_join_and_normalized_ratios(self):
        from plot_metrics_wide import _diagnostic_workload_values

        metric_row = {
            "num_robots": 6.0,
            "mcand": 4.0,
            "noop": 0.25,
            "work_candidate_scan": 10.0,
            "work_active_proposals": 20.0,
        }
        values = _diagnostic_workload_values(metric_row, proposal_mean_ms=12.0)
        self.assertAlmostEqual(values["candidate_entries_per_macro_step"], 24.0, places=8)
        self.assertAlmostEqual(values["active_robot_decisions_per_macro_step"], 4.5, places=8)
        self.assertAlmostEqual(values["proposal_ms_per_candidate_entry"], 0.5, places=8)
        self.assertAlmostEqual(values["proposal_ms_per_active_robot_decision"], 12.0 / 4.5, places=8)

    def test_missing_workload_values_remain_missing(self):
        from plot_metrics_wide import _diagnostic_workload_values

        values = _diagnostic_workload_values({"num_robots": 6.0}, proposal_mean_ms=12.0)
        self.assertIsNone(values["candidate_entries_per_macro_step"])
        self.assertIsNone(values["proposal_ms_per_candidate_entry"])

    def test_matched_seed_episode_intersection(self):
        from plot_metrics_wide import _build_proposal_resolver_diagnostics

        rows = [
            self._canonical_row(measured=10, proposal_total=100, proposal_mean=10, resolution_total=50, resolution_mean=5, seed=1, episode=1, proposer="predicted_reward", resolver="capacity", source="a.csv"),
            self._canonical_row(measured=10, proposal_total=120, proposal_mean=12, resolution_total=55, resolution_mean=5.5, seed=2, episode=1, proposer="predicted_reward", resolver="capacity", source="a.csv"),
            self._canonical_row(measured=10, proposal_total=80, proposal_mean=8, resolution_total=40, resolution_mean=4, seed=1, episode=1, proposer="predicted_reward", resolver="closest_then_capacity", source="b.csv"),
        ]
        timing_groups = [
            {"scenario": "randdest", "route": "nr", "protocol": "aa", "method": "baseline", "inference_mode": "na", "proposer": "predicted_reward", "resolver": "capacity"},
            {"scenario": "randdest", "route": "nr", "protocol": "aa", "method": "baseline", "inference_mode": "na", "proposer": "predicted_reward", "resolver": "closest_then_capacity"},
        ]
        joined_rows = [
            {
                "scenario": "randdest",
                "route_construction": "nearest",
                "protocol": "admission",
                "pol": "predicted_reward",
                "resolver": "capacity",
                "timing_method": "baseline",
                "timing_inference_mode": "na",
                "timing_proposer": "predicted_reward",
                "timing_resolver": "capacity",
            },
            {
                "scenario": "randdest",
                "route_construction": "nearest",
                "protocol": "admission",
                "pol": "predicted_reward",
                "resolver": "closest_then_capacity",
                "timing_method": "baseline",
                "timing_inference_mode": "na",
                "timing_proposer": "predicted_reward",
                "timing_resolver": "closest_then_capacity",
            },
        ]
        _, _, matched_rows, matched_summary, _ = _build_proposal_resolver_diagnostics(
            canonical_rows_all=rows,
            timing_groups_for_combo=timing_groups,
            joined_rows=joined_rows,
        )
        # only (seed=1, episode=1) is common across both resolvers
        self.assertEqual(len(matched_rows), 2)
        self.assertEqual(len(matched_summary), 2)


class PlotMetricsWideTimeCliSmokeTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.python = sys.executable

    def _run(self, args, cwd):
        proc = subprocess.run(
            [self.python, *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return proc.returncode, proc.stdout

    def test_cli_smoke_time_outputs_and_non_time_regression(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            metrics_csv = tdp / "metrics_wide.csv"
            timing_csv = tdp / "timing_summary_v2000_ms3000_mwd300_mtd240_cap2_randdest_closest_nr_aa.csv"
            out_dir = tdp / "out"

            with metrics_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "source_file",
                        "scenario",
                        "instance",
                        "protocol",
                        "resolver",
                        "route_construction",
                        "admission_aware",
                        "pol",
                        "rew",
                        "rew_std",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "source_file": "x.log",
                        "scenario": "randdest",
                        "instance": "i",
                        "protocol": "admission",
                        "resolver": "closest_then_capacity",
                        "route_construction": "nearest",
                        "admission_aware": "1",
                        "pol": "predicted_reward",
                        "rew": "10.0",
                        "rew_std": "0.5",
                    }
                )

            timing_headers = [
                "method",
                "policy",
                "proposer",
                "resolver",
                "protocol",
                "inference_mode",
                "seed",
                "episode",
                "device",
                "timing_protocol",
                "warmup_decisions",
                "measured_decisions",
                "proposal_total_ms",
                "proposal_mean_ms",
                "resolution_total_ms",
                "resolution_mean_ms",
                "env_pre_controller_total_ms",
                "env_pre_controller_mean_ms",
                "pre_step_sync_total_ms",
                "pre_step_sync_mean_ms",
                "commit_dispatch_total_ms",
                "commit_dispatch_mean_ms",
                "simulation_total_ms",
                "simulation_mean_ms",
                "post_step_logging_total_ms",
                "post_step_logging_mean_ms",
                "other_total_ms",
                "other_mean_ms",
                "decision_total_ms",
                "decision_mean_ms",
                "scenario",
                "torch_num_threads",
                "omp_num_threads",
                "mkl_num_threads",
                "host_name",
                "cpu_model",
            ]
            with timing_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=timing_headers)
                writer.writeheader()
                writer.writerow(
                    {
                        "method": "baseline",
                        "policy": "predicted_reward",
                        "proposer": "predicted_reward",
                        "resolver": "closest",
                        "protocol": "admission_aware",
                        "inference_mode": "na",
                        "seed": "1",
                        "episode": "1",
                        "device": "cpu",
                        "timing_protocol": "shared_outer_v1",
                        "warmup_decisions": "0",
                        "measured_decisions": "10",
                        "proposal_total_ms": "100",
                        "proposal_mean_ms": "10",
                        "resolution_total_ms": "50",
                        "resolution_mean_ms": "5",
                        "env_pre_controller_total_ms": "20",
                        "env_pre_controller_mean_ms": "2",
                        "pre_step_sync_total_ms": "10",
                        "pre_step_sync_mean_ms": "1",
                        "commit_dispatch_total_ms": "10",
                        "commit_dispatch_mean_ms": "1",
                        "simulation_total_ms": "200",
                        "simulation_mean_ms": "20",
                        "post_step_logging_total_ms": "10",
                        "post_step_logging_mean_ms": "1",
                        "other_total_ms": "20",
                        "other_mean_ms": "2",
                        "decision_total_ms": "420",
                        "decision_mean_ms": "42",
                        "scenario": "configs/randdest.sumocfg",
                        "torch_num_threads": "1",
                        "omp_num_threads": "1",
                        "mkl_num_threads": "1",
                        "host_name": "h",
                        "cpu_model": "cpu",
                    }
                )

            code, out = self._run(
                [
                    "plot_metrics_wide.py",
                    str(metrics_csv),
                    "--time",
                    "--timing-files",
                    str(timing_csv),
                    "--scenario",
                    "randdest",
                    "--metrics",
                    "rew",
                    "--pareto",
                    "--annotate-time",
                    "--output-dir",
                    str(out_dir),
                ],
                cwd=self.repo_root,
            )
            self.assertEqual(code, 0, msg=out)

            target = out_dir / "randdest" / "time_cmp" / "nr_aa"
            self.assertTrue((target / "rew_time_cmp.png").exists())
            self.assertTrue((target / "proposal_resolution_time_cmp.png").exists())
            self.assertTrue((target / "proposal_time_by_resolver.png").exists())
            self.assertTrue((target / "resolution_time_by_proposer.png").exists())
            self.assertTrue((target / "decision_phase_breakdown_by_combination.png").exists())
            self.assertTrue((target / "decision_phase_breakdown_overall.png").exists())
            self.assertTrue((target / "decision_phase_totals_overall.png").exists())
            self.assertTrue((target / "time_cmp_data.csv").exists())
            self.assertTrue((target / "time_phase_data.csv").exists())

            code2, out2 = self._run(
                [
                    "plot_metrics_wide.py",
                    str(metrics_csv),
                    "--resolver-cmp",
                    "--grouped",
                    "false",
                    "--scenario",
                    "randdest",
                    "--metrics",
                    "rew",
                    "--output-dir",
                    str(out_dir / "base"),
                ],
                cwd=self.repo_root,
            )
            self.assertEqual(code2, 0, msg=out2)
            self.assertTrue((out_dir / "base" / "randdest" / "resolver_cmp" / "nr_aa" / "rew_resolver_cmp.png").exists())

    def test_cli_smoke_time_diagnostics_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            metrics_csv = tdp / "metrics_wide.csv"
            timing_csv_a = tdp / "timing_summary_randdest_capacity_nr_aa.csv"
            timing_csv_b = tdp / "timing_summary_randdest_closest_nr_aa.csv"
            out_dir = tdp / "out"

            with metrics_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "source_file",
                        "scenario",
                        "instance",
                        "protocol",
                        "resolver",
                        "route_construction",
                        "admission_aware",
                        "pol",
                        "rew",
                        "rew_std",
                        "num_robots",
                        "mcand",
                        "noop",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "source_file": "x1.log",
                        "scenario": "randdest",
                        "instance": "i",
                        "protocol": "admission",
                        "resolver": "capacity",
                        "route_construction": "nearest",
                        "admission_aware": "1",
                        "pol": "predicted_reward",
                        "rew": "10.0",
                        "rew_std": "0.5",
                        "num_robots": "6",
                        "mcand": "4",
                        "noop": "0.2",
                    }
                )
                writer.writerow(
                    {
                        "source_file": "x2.log",
                        "scenario": "randdest",
                        "instance": "i",
                        "protocol": "admission",
                        "resolver": "closest_then_capacity",
                        "route_construction": "nearest",
                        "admission_aware": "1",
                        "pol": "predicted_reward",
                        "rew": "11.0",
                        "rew_std": "0.6",
                        "num_robots": "6",
                        "mcand": "3",
                        "noop": "0.1",
                    }
                )

            timing_headers = [
                "method",
                "policy",
                "proposer",
                "resolver",
                "protocol",
                "inference_mode",
                "seed",
                "episode",
                "device",
                "timing_protocol",
                "warmup_decisions",
                "measured_decisions",
                "proposal_total_ms",
                "proposal_mean_ms",
                "proposal_p50_ms",
                "proposal_p90_ms",
                "proposal_p95_ms",
                "proposal_max_ms",
                "resolution_total_ms",
                "resolution_mean_ms",
                "decision_total_ms",
                "decision_mean_ms",
                "scenario",
                "torch_num_threads",
                "omp_num_threads",
                "mkl_num_threads",
                "host_name",
                "cpu_model",
            ]

            def write_timing(path: Path, resolver_name: str, prop_means: list[float]):
                with path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=timing_headers)
                    writer.writeheader()
                    for i, mean_v in enumerate(prop_means, start=1):
                        measured = 10
                        writer.writerow(
                            {
                                "method": "baseline",
                                "policy": "predicted_reward",
                                "proposer": "predicted_reward",
                                "resolver": resolver_name,
                                "protocol": "admission",
                                "inference_mode": "na",
                                "seed": str(i),
                                "episode": "1",
                                "device": "cpu",
                                "timing_protocol": "shared_outer_v1",
                                "warmup_decisions": "0",
                                "measured_decisions": str(measured),
                                "proposal_total_ms": str(mean_v * measured),
                                "proposal_mean_ms": str(mean_v),
                                "proposal_p50_ms": str(mean_v),
                                "proposal_p90_ms": str(mean_v + 0.1),
                                "proposal_p95_ms": str(mean_v + 0.2),
                                "proposal_max_ms": str(mean_v + 0.3),
                                "resolution_total_ms": "20",
                                "resolution_mean_ms": "2",
                                "decision_total_ms": "150",
                                "decision_mean_ms": "15",
                                "scenario": "randdest",
                                "torch_num_threads": "1",
                                "omp_num_threads": "1",
                                "mkl_num_threads": "1",
                                "host_name": "h",
                                "cpu_model": "cpu",
                            }
                        )

            write_timing(timing_csv_a, "capacity", [10.0, 12.0])
            write_timing(timing_csv_b, "closest", [8.0, 9.0])

            code, out = self._run(
                [
                    "plot_metrics_wide.py",
                    str(metrics_csv),
                    "--time",
                    "--time-diagnose-proposer-resolver",
                    "--timing-files",
                    str(timing_csv_a),
                    str(timing_csv_b),
                    "--scenario",
                    "randdest",
                    "--metrics",
                    "rew",
                    "--output-dir",
                    str(out_dir),
                ],
                cwd=self.repo_root,
            )
            self.assertEqual(code, 0, msg=out)

            target = out_dir / "randdest" / "time_cmp" / "nr_aa"
            self.assertTrue((target / "proposal_resolver_diagnostics_runs.csv").exists())
            self.assertTrue((target / "proposal_resolver_diagnostics_groups.csv").exists())
            self.assertTrue((target / "proposal_resolver_matched_replicates.csv").exists())
            self.assertTrue((target / "proposal_resolver_matched_summary.csv").exists())
            self.assertTrue((target / "proposal_latency_by_proposer_and_resolver.png").exists())
            self.assertTrue((target / "proposal_workload_by_proposer_and_resolver.png").exists())
            self.assertTrue((target / "proposal_latency_normalized_by_workload.png").exists())
            self.assertTrue((target / "proposal_latency_vs_candidate_workload.png").exists())
            self.assertTrue((target / "proposal_latency_matched_replicates.png").exists())
            self.assertTrue((target / "proposal_latency_resolver_spread.png").exists())


if __name__ == "__main__":
    unittest.main()
