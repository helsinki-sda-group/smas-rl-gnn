import math
import unittest
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import types as py_types

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SUMO_PKG_PATH = _REPO_ROOT / "sumo_rl_rs"


if "sumo_rl_rs" not in sys.modules:
    pkg = py_types.ModuleType("sumo_rl_rs")
    pkg.__path__ = [str(_SUMO_PKG_PATH)]
    sys.modules["sumo_rl_rs"] = pkg

if "sumo_rl_rs.logging" not in sys.modules:
    logging_pkg = py_types.ModuleType("sumo_rl_rs.logging")
    logging_pkg.__path__ = [str(_SUMO_PKG_PATH / "logging")]
    sys.modules["sumo_rl_rs.logging"] = logging_pkg

if "sumo_rl_rs.logging.ridepool_logger" not in sys.modules:
    logger_mod = py_types.ModuleType("sumo_rl_rs.logging.ridepool_logger")

    class _RidepoolLogger:
        pass

    logger_mod.RidepoolLogger = _RidepoolLogger
    sys.modules["sumo_rl_rs.logging.ridepool_logger"] = logger_mod


_CONTROLLER_PATH = _SUMO_PKG_PATH / "environment" / "rl_controller_adapter.py"
_SPEC = importlib.util.spec_from_file_location("resolver_controller", _CONTROLLER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load controller module from {_CONTROLLER_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

RLControllerAdapter = _MODULE.RLControllerAdapter
Task = _MODULE.Task
ScoredCandidate = _MODULE.ScoredCandidate


class _FakeVehicle:
    def __init__(self, edge_by_robot):
        self._edge_by_robot = dict(edge_by_robot)

    def getRoute(self, rid):
        return [self._edge_by_robot.get(str(rid), "")]


class _FakeSumo:
    def __init__(self, edge_by_robot):
        self.vehicle = _FakeVehicle(edge_by_robot)


class _FakeLogger:
    def __init__(self):
        self.debug_rows = []
        self.centralized_rows = []

    def log_conflict_task_count(self, _n):
        return None

    def log_conflict_metrics_event(self, **_kwargs):
        return None

    def log_conflict(self, *_args, **_kwargs):
        return None

    def log_debug(self, _t, _tag, payload):
        self.debug_rows.append(dict(payload))

    def log_centralized_matching(self, **kwargs):
        self.centralized_rows.append(dict(kwargs))


class RewardResolverTests(unittest.TestCase):
    def _task(self, tid="A", from_edge="task_A"):
        return Task(
            id=str(tid),
            fromEdge=str(from_edge),
            toEdge=f"{from_edge}_to",
            state=RLControllerAdapter.STATE_REQUESTED,
            reservationTime=0.0,
            estTravelTime=10.0,
            pickupDeadline=1000.0,
            dropoffDeadline=2000.0,
            is_obsolete=False,
            is_assigned=False,
        )

    def _adapter(self, mode="predicted_reward", score_map=None, admission_aware=None):
        score_map = dict(score_map or {})
        adapter = object.__new__(RLControllerAdapter)
        adapter.conflict_resolution = mode
        if admission_aware is not None:
            adapter.admission_aware = bool(admission_aware)
        adapter._rng = np.random.default_rng(123)
        adapter._warned_missing_logit_diff = False
        adapter._last_resolver_selected_plans = {}
        adapter._shadow_plan_by_robot = {"r0": ["base"], "r1": ["base"], "r2": ["base"]}
        adapter._step_count = 0
        adapter.logger = _FakeLogger()
        adapter.sumo = _FakeSumo({"r0": "edge_r0", "r1": "edge_r1", "r2": "edge_r2"})

        def _now(self):
            return 0.0

        def _remaining_capacity(self, rid):
            _ = rid
            return 2

        def _reservation_index(self):
            return {
                "A": SimpleNamespace(id="A", state=RLControllerAdapter.STATE_REQUESTED),
                "B": SimpleNamespace(id="B", state=RLControllerAdapter.STATE_REQUESTED),
            }

        def _person_to_res_index(self, _res_index):
            return {}

        def _current_reservation_ids_onboard(self, rid, p2r):
            _ = (rid, p2r)
            return []

        def _road_distance(self, from_edge, to_edge):
            _ = (from_edge, to_edge)
            return 10.0

        call_modes = []

        def _score(self, *, mode, rid, feasible, res_index, current_edge, current_customer_ids, base_score_cache=None):
            _ = (feasible, res_index, current_edge, current_customer_ids, base_score_cache)
            call_modes.append(mode)
            utility, plan = score_map.get((rid, mode), (float("-inf"), None))
            return [
                ScoredCandidate(
                    robot_id=str(rid),
                    task_id="A",
                    task_index=0,
                    feasible=math.isfinite(float(utility)),
                    utility=float(utility),
                    raw_score=float(utility),
                    proposed_plan=(list(plan) if plan is not None else None),
                    metadata={},
                )
            ]

        adapter._remaining_capacity = py_types.MethodType(_remaining_capacity, adapter)
        adapter._reservation_index = py_types.MethodType(_reservation_index, adapter)
        adapter._person_to_res_index = py_types.MethodType(_person_to_res_index, adapter)
        adapter._current_reservation_ids_onboard = py_types.MethodType(_current_reservation_ids_onboard, adapter)
        adapter._road_distance = py_types.MethodType(_road_distance, adapter)
        adapter._score_feasible_candidates_for_mode = py_types.MethodType(_score, adapter)
        adapter._now = py_types.MethodType(_now, adapter)
        adapter._score_calls = call_modes
        return adapter

    def test_non_conflicting_proposal_is_kept(self):
        adapter = self._adapter(mode="predicted_reward", score_map={("r0", "predicted_reward"): (1.0, ["A", "A"])})
        tasks = [self._task("A")]
        resolved, winners = adapter._resolve_assignment_conflicts(["r0", "r1"], ["A", None], tasks)
        self.assertEqual(resolved, ["A", None])
        self.assertEqual(winners, {"A": "r0"})

    def test_predicted_reward_conflict_picks_max_score(self):
        adapter = self._adapter(
            mode="predicted_reward",
            score_map={
                ("r0", "predicted_reward"): (1.0, ["base", "A", "A"]),
                ("r1", "predicted_reward"): (3.0, ["base", "A", "A"]),
            },
        )
        tasks = [self._task("A")]
        resolved, winners = adapter._resolve_assignment_conflicts(["r0", "r1"], ["A", "A"], tasks)
        self.assertEqual(winners["A"], "r1")
        self.assertEqual(resolved, [None, "A"])
        self.assertEqual(adapter._last_resolver_selected_plans.get("r1"), ["base", "A", "A"])

    def test_r3_uses_predicted_reward_mode(self):
        adapter = self._adapter(
            mode="predicted_reward",
            score_map={
                ("r0", "predicted_reward"): (2.0, ["A", "A"]),
                ("r1", "predicted_reward"): (1.0, ["A", "A"]),
            },
        )
        _ = adapter._resolve_assignment_conflicts(["r0", "r1"], ["A", "A"], [self._task("A")])
        self.assertIn("predicted_reward", adapter._score_calls)

    def test_r4_uses_predicted_reward_joint_mode(self):
        adapter = self._adapter(
            mode="predicted_reward_joint",
            score_map={
                ("r0", "predicted_reward_joint"): (2.0, ["A", "A"]),
                ("r1", "predicted_reward_joint"): (1.0, ["A", "A"]),
            },
        )
        _ = adapter._resolve_assignment_conflicts(["r0", "r1"], ["A", "A"], [self._task("A")])
        self.assertIn("predicted_reward_joint", adapter._score_calls)

    def test_r3_and_r4_can_pick_different_winners(self):
        tasks = [self._task("A")]
        adapter_r3 = self._adapter(
            mode="predicted_reward",
            score_map={
                ("r0", "predicted_reward"): (5.0, ["A", "A"]),
                ("r1", "predicted_reward"): (4.0, ["A", "A"]),
            },
        )
        _, winners_r3 = adapter_r3._resolve_assignment_conflicts(["r0", "r1"], ["A", "A"], tasks)

        adapter_r4 = self._adapter(
            mode="predicted_reward_joint",
            score_map={
                ("r0", "predicted_reward_joint"): (1.0, ["A", "A"]),
                ("r1", "predicted_reward_joint"): (2.0, ["A", "A"]),
            },
        )
        _, winners_r4 = adapter_r4._resolve_assignment_conflicts(["r0", "r1"], ["A", "A"], tasks)

        self.assertEqual(winners_r3["A"], "r0")
        self.assertEqual(winners_r4["A"], "r1")

    def test_negative_scores_still_choose_winner(self):
        adapter = self._adapter(
            mode="predicted_reward",
            score_map={
                ("r0", "predicted_reward"): (-5.0, ["A", "A"]),
                ("r1", "predicted_reward"): (-2.0, ["A", "A"]),
            },
        )
        _, winners = adapter._resolve_assignment_conflicts(["r0", "r1"], ["A", "A"], [self._task("A")])
        self.assertEqual(winners["A"], "r1")

    def test_explicit_false_matches_forced_negative_winner_behavior(self):
        adapter = self._adapter(
            mode="predicted_reward",
            score_map={
                ("r0", "predicted_reward"): (-5.0, ["A", "A"]),
                ("r1", "predicted_reward"): (-2.0, ["A", "A"]),
            },
            admission_aware=False,
        )
        _, winners = adapter._resolve_assignment_conflicts(["r0", "r1"], ["A", "A"], [self._task("A")])
        self.assertEqual(winners["A"], "r1")

    def test_tie_is_reproducible_with_seed(self):
        score_map = {
            ("r0", "predicted_reward"): (1.0, ["A", "A"]),
            ("r1", "predicted_reward"): (1.0, ["A", "A"]),
        }
        a1 = self._adapter(mode="predicted_reward", score_map=score_map)
        a2 = self._adapter(mode="predicted_reward", score_map=score_map)
        _, w1 = a1._resolve_assignment_conflicts(["r0", "r1"], ["A", "A"], [self._task("A")])
        _, w2 = a2._resolve_assignment_conflicts(["r0", "r1"], ["A", "A"], [self._task("A")])
        self.assertEqual(w1, w2)

    def test_one_nonfinite_score_does_not_block_valid_winner(self):
        adapter = self._adapter(
            mode="predicted_reward_joint",
            score_map={
                ("r0", "predicted_reward_joint"): (float("nan"), ["A", "A"]),
                ("r1", "predicted_reward_joint"): (0.5, ["A", "A"]),
            },
        )
        _, winners = adapter._resolve_assignment_conflicts(["r0", "r1"], ["A", "A"], [self._task("A")])
        self.assertEqual(winners["A"], "r1")

    def test_all_invalid_scores_trigger_random_fallback(self):
        adapter = self._adapter(
            mode="predicted_reward",
            score_map={
                ("r0", "predicted_reward"): (float("nan"), None),
                ("r1", "predicted_reward"): (float("inf"), None),
            },
        )
        _, winners = adapter._resolve_assignment_conflicts(["r0", "r1"], ["A", "A"], [self._task("A")])
        self.assertIn(winners["A"], {"r0", "r1"})

    def test_scoring_does_not_mutate_shadow_plan(self):
        adapter = self._adapter(
            mode="predicted_reward",
            score_map={
                ("r0", "predicted_reward"): (1.0, ["base", "A", "A"]),
                ("r1", "predicted_reward"): (2.0, ["base", "A", "A"]),
            },
        )
        before = {k: list(v) for k, v in adapter._shadow_plan_by_robot.items()}
        _ = adapter._resolve_assignment_conflicts(["r0", "r1"], ["A", "A"], [self._task("A")])
        self.assertEqual(before, adapter._shadow_plan_by_robot)

    def test_admission_singleton_positive_is_accepted(self):
        adapter = self._adapter(
            mode="predicted_reward",
            score_map={("r0", "predicted_reward"): (0.25, ["A", "A"])},
            admission_aware=True,
        )
        resolved, winners = adapter._resolve_assignment_conflicts(["r0"], ["A"], [self._task("A")])
        self.assertEqual(resolved, ["A"])
        self.assertEqual(winners, {"A": "r0"})

    def test_admission_singleton_nonpositive_is_rejected(self):
        adapter = self._adapter(
            mode="predicted_reward",
            score_map={("r0", "predicted_reward"): (0.0, ["A", "A"])},
            admission_aware=True,
        )
        resolved, winners = adapter._resolve_assignment_conflicts(["r0"], ["A"], [self._task("A")])
        self.assertEqual(resolved, [None])
        self.assertEqual(winners, {})

    def test_admission_conflict_all_nonpositive_has_no_winner(self):
        adapter = self._adapter(
            mode="predicted_reward_joint",
            score_map={
                ("r0", "predicted_reward_joint"): (-0.4, ["A", "A"]),
                ("r1", "predicted_reward_joint"): (0.0, ["A", "A"]),
            },
            admission_aware=True,
        )
        resolved, winners = adapter._resolve_assignment_conflicts(["r0", "r1"], ["A", "A"], [self._task("A")])
        self.assertEqual(resolved, [None, None])
        self.assertEqual(winners, {})

    def test_admission_conflict_best_positive_wins(self):
        adapter = self._adapter(
            mode="predicted_reward_joint",
            score_map={
                ("r0", "predicted_reward_joint"): (-0.1, ["A", "A"]),
                ("r1", "predicted_reward_joint"): (0.6, ["A", "A"]),
            },
            admission_aware=True,
        )
        resolved, winners = adapter._resolve_assignment_conflicts(["r0", "r1"], ["A", "A"], [self._task("A")])
        self.assertEqual(resolved, [None, "A"])
        self.assertEqual(winners, {"A": "r1"})


class HungarianResolverTests(unittest.TestCase):
    def _task(self, tid, edge):
        return Task(
            id=str(tid),
            fromEdge=str(edge),
            toEdge=f"{edge}_to",
            state=RLControllerAdapter.STATE_REQUESTED,
            reservationTime=0.0,
            estTravelTime=10.0,
            pickupDeadline=1000.0,
            dropoffDeadline=2000.0,
            is_obsolete=False,
            is_assigned=False,
        )

    def _adapter(self, mode, *, edge_by_robot, distances, score_map, rem_cap=None, vicinity=1000.0, admission_aware=False):
        rem_cap = dict(rem_cap or {})
        adapter = object.__new__(RLControllerAdapter)
        adapter.candidates_sorting = mode
        adapter.admission_aware = bool(admission_aware)
        adapter.vicinity_m = float(vicinity)
        adapter.competition_joint_tie_tolerance = 1e-8
        adapter._rng = np.random.default_rng(77)
        adapter._shadow_plan_by_robot = {}
        adapter.logger = _FakeLogger()
        adapter.sumo = _FakeSumo(edge_by_robot)

        def _now(self):
            return 0.0

        def _reservation_index(self):
            return {
                "task_a": SimpleNamespace(id="task_a", state=RLControllerAdapter.STATE_REQUESTED),
                "task_b": SimpleNamespace(id="task_b", state=RLControllerAdapter.STATE_REQUESTED),
                "task_c": SimpleNamespace(id="task_c", state=RLControllerAdapter.STATE_REQUESTED),
                "task_d": SimpleNamespace(id="task_d", state=RLControllerAdapter.STATE_REQUESTED),
            }

        def _person_to_res_index(self, _res_index):
            return {}

        def _current_reservation_ids_onboard(self, rid, p2r):
            _ = (rid, p2r)
            return []

        def _remaining_capacity(self, rid):
            return int(rem_cap.get(str(rid), 1))

        def _locked_owner_for_reservation(self, res_id, task=None):
            _ = (res_id, task)
            return None

        def _road_distance(self, from_edge, to_edge):
            return float(distances.get((str(from_edge), str(to_edge)), float("inf")))

        def _score(self, *, mode, rid, feasible, res_index, current_edge, current_customer_ids, base_score_cache=None):
            _ = (res_index, current_edge, current_customer_ids, base_score_cache)
            rows = []
            for dist, j, task in feasible:
                val = float(score_map[(str(rid), str(task.id))])
                rows.append(
                    ScoredCandidate(
                        robot_id=str(rid),
                        task_id=str(task.id),
                        task_index=int(j),
                        feasible=math.isfinite(val),
                        utility=val,
                        raw_score=val,
                        proposed_plan=[str(task.id), str(task.id)],
                        metadata={"pickup_distance": float(dist)},
                    )
                )
            return rows

        adapter._now = py_types.MethodType(_now, adapter)
        adapter._reservation_index = py_types.MethodType(_reservation_index, adapter)
        adapter._person_to_res_index = py_types.MethodType(_person_to_res_index, adapter)
        adapter._current_reservation_ids_onboard = py_types.MethodType(_current_reservation_ids_onboard, adapter)
        adapter._remaining_capacity = py_types.MethodType(_remaining_capacity, adapter)
        adapter._locked_owner_for_reservation = py_types.MethodType(_locked_owner_for_reservation, adapter)
        adapter._road_distance = py_types.MethodType(_road_distance, adapter)
        adapter._score_feasible_candidates_for_mode = py_types.MethodType(_score, adapter)
        return adapter

    def test_counterexample_total_score_17(self):
        tasks = [self._task("task_a", "ea"), self._task("task_b", "eb")]
        adapter = self._adapter(
            "predicted_reward",
            edge_by_robot={"robot_1": "r1", "robot_2": "r2"},
            distances={
                ("r1", "ea"): 1.0,
                ("r1", "eb"): 1.0,
                ("r2", "ea"): 1.0,
                ("r2", "eb"): 1.0,
            },
            score_map={
                ("robot_1", "task_a"): 10.0,
                ("robot_1", "task_b"): 9.0,
                ("robot_2", "task_a"): 8.0,
                ("robot_2", "task_b"): 0.0,
            },
        )
        chosen, winners, _plans = adapter._resolve_assignments_hungarian(robots=["robot_1", "robot_2"], tasks_list=tasks)
        self.assertEqual(chosen, ["task_b", "task_a"])
        self.assertEqual(winners, {"task_b": "robot_1", "task_a": "robot_2"})

    def test_sparse_edges_and_zero_candidate_robot(self):
        tasks = [self._task("task_a", "ea"), self._task("task_b", "eb")]
        adapter = self._adapter(
            "predicted_reward",
            edge_by_robot={"r0": "e0", "r1": "e1", "r2": "e2"},
            distances={
                ("e0", "ea"): 5.0,
                ("e1", "eb"): 5.0,
            },
            score_map={
                ("r0", "task_a"): 3.0,
                ("r1", "task_b"): 4.0,
            },
            vicinity=10.0,
        )
        chosen, winners, _ = adapter._resolve_assignments_hungarian(robots=["r0", "r1", "r2"], tasks_list=tasks)
        self.assertEqual(chosen[0], "task_a")
        self.assertEqual(chosen[1], "task_b")
        self.assertIsNone(chosen[2])
        self.assertEqual(len(winners), 2)

    def test_more_robots_than_tasks(self):
        tasks = [self._task("task_a", "ea")]
        adapter = self._adapter(
            "predicted_reward",
            edge_by_robot={"r0": "e0", "r1": "e1"},
            distances={("e0", "ea"): 1.0, ("e1", "ea"): 1.0},
            score_map={("r0", "task_a"): 5.0, ("r1", "task_a"): 4.0},
        )
        chosen, winners, _ = adapter._resolve_assignments_hungarian(robots=["r0", "r1"], tasks_list=tasks)
        self.assertEqual(sum(x is not None for x in chosen), 1)
        self.assertEqual(len(winners), 1)

    def test_negative_scores_still_maximize_cardinality(self):
        tasks = [self._task("task_a", "ea"), self._task("task_b", "eb")]
        adapter = self._adapter(
            "predicted_reward",
            edge_by_robot={"r0": "e0", "r1": "e1"},
            distances={("e0", "ea"): 1.0, ("e0", "eb"): 1.0, ("e1", "ea"): 1.0, ("e1", "eb"): 1.0},
            score_map={
                ("r0", "task_a"): -10.0,
                ("r0", "task_b"): -11.0,
                ("r1", "task_a"): -12.0,
                ("r1", "task_b"): -9.0,
            },
        )
        chosen, winners, _ = adapter._resolve_assignments_hungarian(robots=["r0", "r1"], tasks_list=tasks)
        self.assertEqual(sum(x is not None for x in chosen), 2)
        self.assertEqual(len(winners), 2)

    def test_no_duplicate_robot_or_task_and_infeasible_not_selected(self):
        tasks = [self._task("task_a", "ea"), self._task("task_b", "eb"), self._task("task_c", "ec")]
        adapter = self._adapter(
            "predicted_reward",
            edge_by_robot={"r0": "e0", "r1": "e1", "r2": "e2"},
            distances={
                ("e0", "ea"): 1.0,
                ("e1", "eb"): 1.0,
                ("e2", "ec"): 1.0,
            },
            score_map={
                ("r0", "task_a"): 2.0,
                ("r1", "task_b"): 3.0,
                ("r2", "task_c"): 4.0,
            },
            vicinity=5.0,
        )
        chosen, winners, plans = adapter._resolve_assignments_hungarian(robots=["r0", "r1", "r2"], tasks_list=tasks)
        assigned_tasks = [x for x in chosen if x is not None]
        self.assertEqual(len(assigned_tasks), len(set(assigned_tasks)))
        self.assertEqual(len(winners), len(set(winners.values())))
        self.assertEqual(set(plans.keys()), set(winners.values()))

    def test_admission_hungarian_prefers_dummy_over_negative_edges(self):
        tasks = [self._task("task_a", "ea"), self._task("task_b", "eb")]
        adapter = self._adapter(
            "predicted_reward",
            edge_by_robot={"r0": "e0", "r1": "e1"},
            distances={
                ("e0", "ea"): 1.0,
                ("e0", "eb"): 1.0,
                ("e1", "ea"): 1.0,
                ("e1", "eb"): 1.0,
            },
            score_map={
                ("r0", "task_a"): -0.2,
                ("r0", "task_b"): -0.4,
                ("r1", "task_a"): -0.1,
                ("r1", "task_b"): -0.3,
            },
            admission_aware=True,
        )
        chosen, winners, _ = adapter._resolve_assignments_hungarian(robots=["r0", "r1"], tasks_list=tasks)
        self.assertEqual(chosen, [None, None])
        self.assertEqual(winners, {})

    def test_admission_hungarian_can_leave_multiple_robots_unmatched(self):
        tasks = [self._task("task_a", "ea"), self._task("task_b", "eb")]
        adapter = self._adapter(
            "predicted_reward",
            edge_by_robot={"r0": "e0", "r1": "e1", "r2": "e2"},
            distances={
                ("e0", "ea"): 1.0,
                ("e1", "eb"): 1.0,
                ("e2", "ea"): 1.0,
                ("e2", "eb"): 1.0,
            },
            score_map={
                ("r0", "task_a"): 1.2,
                ("r0", "task_b"): -1.0,
                ("r1", "task_a"): -0.5,
                ("r1", "task_b"): 0.8,
                ("r2", "task_a"): -0.3,
                ("r2", "task_b"): -0.2,
            },
            admission_aware=True,
        )
        chosen, winners, _ = adapter._resolve_assignments_hungarian(robots=["r0", "r1", "r2"], tasks_list=tasks)
        self.assertEqual(chosen, ["task_a", "task_b", None])
        self.assertEqual(winners, {"task_a": "r0", "task_b": "r1"})


class ScoreEquivalenceTests(unittest.TestCase):
    def _task(self, tid, edge, deadline):
        return Task(
            id=str(tid),
            fromEdge=str(edge),
            toEdge=f"{edge}_to",
            state=RLControllerAdapter.STATE_REQUESTED,
            reservationTime=0.0,
            estTravelTime=10.0,
            pickupDeadline=float(deadline),
            dropoffDeadline=2000.0,
            is_obsolete=False,
            is_assigned=False,
        )

    def _adapter(self):
        adapter = object.__new__(RLControllerAdapter)
        adapter._rng = np.random.default_rng(11)
        adapter._shadow_plan_by_robot = {"r0": []}

        def _build_assignment_plan_for_candidate(self, *, rid, task_id, res_index, current_edge, current_customer_ids):
            _ = (rid, res_index, current_edge, current_customer_ids)
            return [task_id, task_id]

        adapter._build_assignment_plan_for_candidate = py_types.MethodType(_build_assignment_plan_for_candidate, adapter)
        return adapter

    def test_distance_deadline_and_deadline_distance_top1_equivalence(self):
        adapter = self._adapter()
        t1 = self._task("a", "ea", 50.0)
        t2 = self._task("b", "eb", 20.0)
        t3 = self._task("c", "ec", 20.0)
        feasible = [
            (30.0, 0, t1),
            (40.0, 1, t2),
            (10.0, 2, t3),
        ]

        scored_dist = adapter._score_feasible_candidates_for_mode(
            mode="pickup_distance",
            rid="r0",
            feasible=feasible,
            res_index={"a": object(), "b": object(), "c": object()},
            current_edge="er",
            current_customer_ids=[],
        )
        best_dist = max(scored_dist, key=lambda x: x.utility)
        self.assertEqual(best_dist.task_id, "c")

        scored_deadline = adapter._score_feasible_candidates_for_mode(
            mode="pickup_deadline",
            rid="r0",
            feasible=feasible,
            res_index={"a": object(), "b": object(), "c": object()},
            current_edge="er",
            current_customer_ids=[],
        )
        best_deadline = max(scored_deadline, key=lambda x: x.utility)
        self.assertEqual(best_deadline.task_id, "c")

        scored_combo = adapter._score_feasible_candidates_for_mode(
            mode="pickup_deadline_distance",
            rid="r0",
            feasible=feasible,
            res_index={"a": object(), "b": object(), "c": object()},
            current_edge="er",
            current_customer_ids=[],
        )
        best_combo = max(scored_combo, key=lambda x: x.utility)
        self.assertEqual(best_combo.task_id, "c")
        self.assertEqual(best_combo.proposed_plan, ["c", "c"])


if __name__ == "__main__":
    unittest.main()
