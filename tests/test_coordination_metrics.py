import os
import tempfile
import unittest

import pandas as pd

from sumo_rl_rs.environment.rl_controller_adapter import compute_coordination_step_counters
from utils.metrics_calculator import compute_episode_metrics_from_logs


class CoordinationMetricsTests(unittest.TestCase):
    def _episode_metrics_from_rows(self, rows):
        with tempfile.TemporaryDirectory() as td:
            coord_path = os.path.join(td, "coordination.csv")
            pd.DataFrame(rows).to_csv(coord_path, index=False)
            return compute_episode_metrics_from_logs(
                episode_dir=td,
                episode_info={},
                policy="test",
                seed=1,
                num_robots=6,
            )

    def test_case1_forced_and_unforced_noop(self):
        counters, off_props, validations = compute_coordination_step_counters(
            robots=["r1", "r2", "r3", "r4", "r5", "r6"],
            has_candidate=[False, False, False, True, True, True],
            selected_noop=[False, False, False, True, True, False],
            proposed_tasks=[None, None, None, None, None, "A"],
            final_assignments=[None, None, None, None, None, "A"],
        )
        self.assertEqual(counters["robot_decisions"], 6)
        self.assertEqual(counters["empty_candidate_decisions"], 3)
        self.assertEqual(counters["nonempty_candidate_decisions"], 3)
        self.assertEqual(counters["unforced_noops"], 2)
        self.assertEqual(counters["real_proposals"], 1)
        self.assertEqual(counters["unique_proposals"], 1)
        self.assertEqual(counters["survived_proposals"], 1)
        self.assertEqual(counters["off_proposal_assignments"], 0)
        self.assertEqual(off_props, [])
        self.assertEqual(validations, [])

        metrics = self._episode_metrics_from_rows([
            {"time": 1.0, **counters},
        ])
        self.assertAlmostEqual(metrics.empty_candidate_rate, 0.5)
        self.assertAlmostEqual(metrics.unforced_noop_rate, 2.0 / 3.0)
        self.assertAlmostEqual(metrics.nonconflicting_proposal_rate, 1.0)
        self.assertAlmostEqual(metrics.proposal_survival_rate, 1.0)

    def test_case2_unique_plus_three_robot_conflict(self):
        counters, _, _ = compute_coordination_step_counters(
            robots=["r1", "r2", "r3", "r4", "r5", "r6"],
            has_candidate=[True, True, True, True, True, True],
            selected_noop=[False, False, False, False, True, True],
            proposed_tasks=["U", "C", "C", "C", None, None],
            final_assignments=["U", None, None, "C", None, None],
        )
        self.assertEqual(counters["real_proposals"], 4)
        self.assertEqual(counters["unique_proposals"], 1)
        self.assertEqual(counters["conflicting_proposals"], 3)
        self.assertEqual(counters["conflict_buckets"], 1)
        self.assertEqual(counters["survived_proposals"], 2)

        metrics = self._episode_metrics_from_rows([
            {"time": 1.0, **counters},
        ])
        self.assertAlmostEqual(metrics.nonconflicting_proposal_rate, 0.25)
        self.assertAlmostEqual(metrics.proposal_survival_rate, 0.5)

    def test_case3_two_conflict_buckets(self):
        counters, _, validations = compute_coordination_step_counters(
            robots=["r1", "r2", "r3", "r4", "r5", "r6"],
            has_candidate=[True, True, True, True, True, True],
            selected_noop=[False, False, False, False, False, False],
            proposed_tasks=["A", "B", "B", "C", "C", "C"],
            final_assignments=["A", "B", None, "C", None, None],
        )
        self.assertEqual(counters["real_proposals"], 6)
        self.assertEqual(counters["unique_proposals"], 1)
        self.assertEqual(counters["conflicting_proposals"], 5)
        self.assertEqual(counters["conflict_buckets"], 2)
        self.assertEqual(counters["survived_proposals"], 3)
        self.assertEqual(validations, [])

        metrics = self._episode_metrics_from_rows([
            {"time": 1.0, **counters},
        ])
        self.assertAlmostEqual(metrics.nonconflicting_proposal_rate, 1.0 / 6.0)
        self.assertAlmostEqual(metrics.proposal_survival_rate, 0.5)

    def test_case4_off_proposal_assignment(self):
        counters, off_props, _ = compute_coordination_step_counters(
            robots=["r1"],
            has_candidate=[True],
            selected_noop=[False],
            proposed_tasks=["A"],
            final_assignments=["B"],
        )
        self.assertEqual(counters["survived_proposals"], 0)
        self.assertEqual(counters["off_proposal_assignments"], 1)
        self.assertEqual(len(off_props), 1)
        self.assertEqual(off_props[0]["robot"], "r1")
        self.assertEqual(off_props[0]["original_proposal"], "A")
        self.assertEqual(off_props[0]["final_assignment"], "B")

    def test_case5_no_real_proposals(self):
        counters, _, _ = compute_coordination_step_counters(
            robots=["r1", "r2", "r3", "r4"],
            has_candidate=[False, False, True, True],
            selected_noop=[False, False, True, True],
            proposed_tasks=[None, None, None, None],
            final_assignments=[None, None, None, None],
        )
        self.assertEqual(counters["real_proposals"], 0)

        metrics = self._episode_metrics_from_rows([
            {"time": 1.0, **counters},
        ])
        self.assertEqual(metrics.nonconflicting_proposal_rate, 0.0)
        self.assertEqual(metrics.proposal_survival_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
