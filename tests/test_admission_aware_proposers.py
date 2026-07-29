import unittest
from types import MethodType, SimpleNamespace

import numpy as np

from sumo_rl_rs.environment.rl_controller_adapter import InsertionEvaluation, RLControllerAdapter, Task


class _FakeVehicle:
    def __init__(self, edge_by_robot):
        self._edge_by_robot = dict(edge_by_robot)

    def getRoute(self, rid):
        return [self._edge_by_robot.get(str(rid), "")]


class _FakeSumo:
    def __init__(self, edge_by_robot):
        self.vehicle = _FakeVehicle(edge_by_robot)


class AdmissionAwareProposerTests(unittest.TestCase):
    def _task(self, tid: str, edge: str) -> Task:
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

    def _adapter(self, *, mode: str, admission_aware: bool, tasks, pred_scores=None, joint_scores=None):
        pred_scores = dict(pred_scores or {})
        joint_scores = dict(joint_scores or {})

        adapter = object.__new__(RLControllerAdapter)
        adapter.k_max = 8
        adapter.vicinity_m = 2_000.0
        adapter.candidates_sorting = str(mode)
        adapter.admission_aware = bool(admission_aware)
        adapter.competition_joint_tie_tolerance = 1e-8
        adapter.competition_joint_competitors = "two_hop"
        adapter._last_robot_ids = ["r0", "r1"]
        adapter._rng = np.random.default_rng(5)
        adapter._shadow_plan_by_robot = {"r0": [], "r1": []}
        adapter._last_competition_joint_diag = {}
        adapter.logger = None
        adapter.sumo = _FakeSumo({"r0": "edge_r0", "r1": "edge_r1"})

        def _reservation_index(self):
            out = {}
            for task in tasks:
                out[str(task.id)] = SimpleNamespace(id=str(task.id), state=RLControllerAdapter.STATE_REQUESTED)
            return out

        def _person_to_res_index(self, _res_index):
            return {}

        def _current_reservation_ids_onboard(self, rid, p2r):
            _ = (rid, p2r)
            return []

        def _remaining_capacity(self, rid):
            _ = rid
            return 2

        def _locked_owner_for_reservation(self, res_id, task=None):
            _ = (res_id, task)
            return None

        def _road_distance(self, from_edge, to_edge):
            _ = from_edge
            return 100.0 if to_edge else float("inf")

        def _is_completed(self, res_obj):
            _ = res_obj
            return False

        def _get_tasks(self):
            return list(tasks)

        def _estimate_sequence_candidate_prediction(self, *, rid, candidate, res_index, current_edge=None, current_customer_ids=None, return_sequence=False):
            _ = (res_index, current_edge, current_customer_ids)
            score = float(pred_scores.get((str(rid), str(candidate.id)), float("-inf")))
            pickup_time = 10.0
            pickup_distance = 100.0
            seq = [str(candidate.id), str(candidate.id)]
            if return_sequence:
                return score, pickup_time, pickup_distance, seq
            return score, pickup_time, pickup_distance

        def _evaluate_marginal_insertion(self, *, robot_id, task, res_index, current_edge, current_customer_ids, base_score_cache=None):
            _ = (res_index, current_edge, current_customer_ids, base_score_cache)
            score = float(joint_scores.get((str(robot_id), str(task.id)), float("-inf")))
            feasible = np.isfinite(score)
            return InsertionEvaluation(
                feasible=bool(feasible),
                base_route_score=0.0,
                inserted_route_score=score,
                marginal_score=score,
                candidate_pickup_time=10.0,
                pickup_distance=100.0,
            )

        adapter._reservation_index = MethodType(_reservation_index, adapter)
        adapter._person_to_res_index = MethodType(_person_to_res_index, adapter)
        adapter._current_reservation_ids_onboard = MethodType(_current_reservation_ids_onboard, adapter)
        adapter._remaining_capacity = MethodType(_remaining_capacity, adapter)
        adapter._locked_owner_for_reservation = MethodType(_locked_owner_for_reservation, adapter)
        adapter._road_distance = MethodType(_road_distance, adapter)
        adapter._is_completed = MethodType(_is_completed, adapter)
        adapter.get_tasks = MethodType(_get_tasks, adapter)
        adapter._estimate_sequence_candidate_prediction = MethodType(_estimate_sequence_candidate_prediction, adapter)
        adapter.evaluate_marginal_insertion = MethodType(_evaluate_marginal_insertion, adapter)
        return adapter

    def test_predicted_reward_admission_positive_keeps_candidate(self):
        tasks = [self._task("A", "task_A")]
        adapter = self._adapter(
            mode="predicted_reward",
            admission_aware=True,
            tasks=tasks,
            pred_scores={("r0", "A"): 0.5},
        )
        returned_tasks, cand_lists = adapter.get_tasks_and_candidate_lists()
        self.assertEqual([returned_tasks[idx].id for idx in cand_lists[0]], ["A"])

    def test_predicted_reward_admission_nonpositive_returns_noop(self):
        tasks = [self._task("A", "task_A")]
        adapter = self._adapter(
            mode="predicted_reward",
            admission_aware=True,
            tasks=tasks,
            pred_scores={("r0", "A"): 0.0},
        )
        _returned_tasks, cand_lists = adapter.get_tasks_and_candidate_lists()
        self.assertEqual(cand_lists[0], [])

    def test_predicted_reward_forced_mode_still_keeps_best_negative_candidate(self):
        tasks = [self._task("A", "task_A"), self._task("B", "task_B")]
        adapter = self._adapter(
            mode="predicted_reward",
            admission_aware=False,
            tasks=tasks,
            pred_scores={("r0", "A"): -1.0, ("r0", "B"): -0.2},
        )
        returned_tasks, cand_lists = adapter.get_tasks_and_candidate_lists()
        self.assertEqual([returned_tasks[idx].id for idx in cand_lists[0]], ["B", "A"])

    def test_proposal_joint_competition_admission_uses_joint_marginal_for_noop(self):
        tasks = [self._task("A", "task_A")]
        adapter = self._adapter(
            mode="proposal_joint_competition",
            admission_aware=True,
            tasks=tasks,
            joint_scores={("r0", "A"): -0.01, ("r1", "A"): -0.5},
        )
        _returned_tasks, cand_lists = adapter.get_tasks_and_candidate_lists()
        self.assertEqual(cand_lists[0], [])


if __name__ == "__main__":
    unittest.main()
