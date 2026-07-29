import unittest
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import types as py_types


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
_SPEC = importlib.util.spec_from_file_location("competition_joint_controller", _CONTROLLER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load controller module from {_CONTROLLER_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

RLControllerAdapter = _MODULE.RLControllerAdapter
Task = _MODULE.Task
InsertionEvaluation = _MODULE.InsertionEvaluation


class _FakeVehicle:
    def __init__(self, edge_by_robot):
        self._edge_by_robot = dict(edge_by_robot)

    def getRoute(self, rid):
        return [self._edge_by_robot.get(str(rid), "")]


class _FakeSumo:
    def __init__(self, edge_by_robot):
        self.vehicle = _FakeVehicle(edge_by_robot)


class CompetitionJointProposerTests(unittest.TestCase):
    def _make_task(self, tid, edge):
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

    def _make_comp_adapter(self, *, mode, tasks, score_map, distance_map, tie_tol=1e-8, vicinity=2000.0):
        adapter = object.__new__(RLControllerAdapter)
        adapter.k_max = 8
        adapter.vicinity_m = float(vicinity)
        adapter.candidates_sorting = str(mode)
        adapter._last_robot_ids = ["r0", "r1"]
        adapter._rng = __import__("numpy").random.default_rng(123)
        adapter.logger = None
        adapter._last_cand_lists = []
        adapter._last_task_ids = []
        adapter._last_competition_joint_diag = {}
        adapter.competition_joint_tie_tolerance = float(tie_tol)
        adapter.competition_joint_competitors = "two_hop"
        adapter._shadow_plan_by_robot = {"r0": ["base_r0"], "r1": ["base_r1"]}

        adapter.sumo = _FakeSumo({"r0": "edge_r0", "r1": "edge_r1"})

        def _reservation_index(self):
            out = {
                "base_r0": SimpleNamespace(id="base_r0", state=RLControllerAdapter.STATE_REQUESTED),
                "base_r1": SimpleNamespace(id="base_r1", state=RLControllerAdapter.STATE_REQUESTED),
            }
            for t in tasks:
                out[str(t.id)] = SimpleNamespace(id=str(t.id), state=RLControllerAdapter.STATE_REQUESTED)
            return out

        def _person_to_res_index(self, _res_index):
            return {}

        def _current_reservation_ids_onboard(self, rid, p2r):
            _ = p2r
            return []

        def _remaining_capacity(self, rid):
            _ = rid
            return 2

        def _locked_owner_for_reservation(self, res_id, task=None):
            _ = (res_id, task)
            return None

        def _road_distance(self, from_edge, to_edge):
            return float(distance_map.get((str(from_edge), str(to_edge)), 1e9))

        def _is_completed(self, res_obj):
            _ = res_obj
            return False

        def _get_tasks(self):
            return list(tasks)

        def _evaluate(self, *, robot_id, task, res_index, current_edge, current_customer_ids, base_score_cache=None):
            _ = (res_index, current_edge, current_customer_ids, base_score_cache)
            key = (str(robot_id), str(task.id))
            if key not in score_map:
                return InsertionEvaluation(
                    feasible=False,
                    base_route_score=float("-inf"),
                    inserted_route_score=float("-inf"),
                    marginal_score=float("-inf"),
                    candidate_pickup_time=float("inf"),
                    pickup_distance=float("inf"),
                )
            score, after_score, pickup_time, pickup_distance, feasible = score_map[key]
            return InsertionEvaluation(
                feasible=bool(feasible),
                base_route_score=float(after_score - score),
                inserted_route_score=float(after_score),
                marginal_score=float(score),
                candidate_pickup_time=float(pickup_time),
                pickup_distance=float(pickup_distance),
            )

        adapter._reservation_index = py_types.MethodType(_reservation_index, adapter)
        adapter._person_to_res_index = py_types.MethodType(_person_to_res_index, adapter)
        adapter._current_reservation_ids_onboard = py_types.MethodType(_current_reservation_ids_onboard, adapter)
        adapter._remaining_capacity = py_types.MethodType(_remaining_capacity, adapter)
        adapter._locked_owner_for_reservation = py_types.MethodType(_locked_owner_for_reservation, adapter)
        adapter._road_distance = py_types.MethodType(_road_distance, adapter)
        adapter._is_completed = py_types.MethodType(_is_completed, adapter)
        adapter.get_tasks = py_types.MethodType(_get_tasks, adapter)
        adapter.evaluate_marginal_insertion = py_types.MethodType(_evaluate, adapter)
        return adapter

    def _cand_ids_for_robot(self, adapter, robot_index=0):
        tasks, cand_lists = adapter.get_tasks_and_candidate_lists()
        return [tasks[idx].id for idx in cand_lists[robot_index]]

    def test_1_ego_clearly_best(self):
        tasks = [self._make_task("A", "task_A")]
        dist = {
            ("edge_r0", "task_A"): 100.0,
            ("edge_r1", "task_A"): 100.0,
        }
        scores = {
            ("r0", "A"): (5.0, 7.0, 10.0, 100.0, True),
            ("r1", "A"): (2.0, 4.0, 12.0, 100.0, True),
        }
        adapter = self._make_comp_adapter(mode="predicted_reward_joint_competition", tasks=tasks, score_map=scores, distance_map=dist)
        self.assertEqual(self._cand_ids_for_robot(adapter), ["A"])

    def test_2_competitor_clearly_best(self):
        tasks = [self._make_task("A", "task_A")]
        dist = {
            ("edge_r0", "task_A"): 100.0,
            ("edge_r1", "task_A"): 100.0,
        }
        scores = {
            ("r0", "A"): (4.0, 6.0, 10.0, 100.0, True),
            ("r1", "A"): (6.0, 8.0, 9.0, 95.0, True),
        }
        adapter = self._make_comp_adapter(mode="predicted_reward_joint_competition", tasks=tasks, score_map=scores, distance_map=dist)
        self.assertEqual(self._cand_ids_for_robot(adapter), [])

    def test_3_tie_within_tolerance(self):
        tasks = [self._make_task("A", "task_A")]
        dist = {
            ("edge_r0", "task_A"): 100.0,
            ("edge_r1", "task_A"): 100.0,
        }
        scores = {
            ("r0", "A"): (1.0, 2.0, 10.0, 100.0, True),
            ("r1", "A"): (1.0 + 5e-9, 2.0, 10.0, 100.0, True),
        }
        adapter = self._make_comp_adapter(
            mode="predicted_reward_joint_competition",
            tasks=tasks,
            score_map=scores,
            distance_map=dist,
            tie_tol=1e-8,
        )
        self.assertEqual(self._cand_ids_for_robot(adapter), ["A"])

    def test_4_infeasible_competitor(self):
        tasks = [self._make_task("A", "task_A")]
        dist = {
            ("edge_r0", "task_A"): 100.0,
            ("edge_r1", "task_A"): 100.0,
        }
        scores = {
            ("r0", "A"): (3.0, 4.0, 10.0, 100.0, True),
            ("r1", "A"): (-999.0, -999.0, float("inf"), float("inf"), False),
        }
        adapter = self._make_comp_adapter(mode="predicted_reward_joint_competition", tasks=tasks, score_map=scores, distance_map=dist)
        self.assertEqual(self._cand_ids_for_robot(adapter), ["A"])

    def test_5_no_competitors_equivalent_to_joint(self):
        tasks = [self._make_task("A", "task_A")]
        dist = {
            ("edge_r0", "task_A"): 100.0,
            ("edge_r1", "task_A"): 5000.0,
        }
        scores = {
            ("r0", "A"): (3.5, 5.0, 10.0, 100.0, True),
        }
        comp_adapter = self._make_comp_adapter(mode="predicted_reward_joint_competition", tasks=tasks, score_map=scores, distance_map=dist)
        joint_adapter = self._make_comp_adapter(mode="predicted_reward_joint", tasks=tasks, score_map=scores, distance_map=dist)
        self.assertEqual(self._cand_ids_for_robot(comp_adapter), self._cand_ids_for_robot(joint_adapter))

    def test_6_multiple_tasks_best_owned_pick_highest(self):
        tasks = [self._make_task("A", "task_A"), self._make_task("B", "task_B")]
        dist = {
            ("edge_r0", "task_A"): 100.0,
            ("edge_r1", "task_A"): 100.0,
            ("edge_r0", "task_B"): 120.0,
            ("edge_r1", "task_B"): 120.0,
        }
        scores = {
            ("r0", "A"): (4.0, 6.0, 12.0, 120.0, True),
            ("r1", "A"): (2.0, 3.0, 13.0, 120.0, True),
            ("r0", "B"): (5.0, 7.0, 14.0, 120.0, True),
            ("r1", "B"): (1.0, 2.0, 15.0, 120.0, True),
        }
        adapter = self._make_comp_adapter(mode="predicted_reward_joint_competition", tasks=tasks, score_map=scores, distance_map=dist)
        self.assertEqual(self._cand_ids_for_robot(adapter)[0], "B")

    def test_7_all_tasks_suppressed_returns_noop(self):
        tasks = [self._make_task("A", "task_A"), self._make_task("B", "task_B")]
        dist = {
            ("edge_r0", "task_A"): 100.0,
            ("edge_r1", "task_A"): 100.0,
            ("edge_r0", "task_B"): 120.0,
            ("edge_r1", "task_B"): 120.0,
        }
        scores = {
            ("r0", "A"): (1.0, 3.0, 10.0, 100.0, True),
            ("r1", "A"): (2.0, 4.0, 9.0, 100.0, True),
            ("r0", "B"): (1.5, 3.5, 10.0, 120.0, True),
            ("r1", "B"): (2.5, 4.5, 9.0, 120.0, True),
        }
        adapter = self._make_comp_adapter(mode="predicted_reward_joint_competition", tasks=tasks, score_map=scores, distance_map=dist)
        self.assertEqual(self._cand_ids_for_robot(adapter), [])

    def test_8_side_effect_safety(self):
        adapter = object.__new__(RLControllerAdapter)
        adapter._shadow_plan_by_robot = {"r0": ["base"]}
        adapter._task_lifecycle = {}
        adapter.reward_weights = {"comp": 3.0, "wait": 1.5, "travel": 1.25}
        adapter.reward_caps = {"wait": 600.0, "travel": 240.0}

        def _is_completed(self, res_obj):
            _ = res_obj
            return False

        def _is_picked(self, res_obj):
            _ = res_obj
            return False

        def _task_from_reservation(self, res_id, res_obj):
            _ = res_obj
            return Task(
                id=str(res_id),
                fromEdge="x",
                toEdge="y",
                state=RLControllerAdapter.STATE_REQUESTED,
                reservationTime=0.0,
                estTravelTime=10.0,
                pickupDeadline=1000.0,
                dropoffDeadline=2000.0,
            )

        def _estimate_plan_event_times(self, *, rid, res_ids, res_index, current_edge, current_customer_ids):
            _ = (rid, res_index, current_edge, current_customer_ids)
            out = {}
            t = 10.0
            for res_id in res_ids:
                out[str(res_id)] = {"pickup": t, "dropoff": t + 10.0}
                t += 1.0
            return out

        def _task_score(self, *, task, predicted_pickup_time, predicted_dropoff_time, actual_pickup_time, preserve_actual_pickup_validity):
            _ = (task, actual_pickup_time, preserve_actual_pickup_validity)
            return float(predicted_dropoff_time - predicted_pickup_time)

        def _road_distance(self, from_edge, to_edge):
            _ = (from_edge, to_edge)
            return 42.0

        adapter._is_completed = py_types.MethodType(_is_completed, adapter)
        adapter._is_picked = py_types.MethodType(_is_picked, adapter)
        adapter._task_from_reservation = py_types.MethodType(_task_from_reservation, adapter)
        adapter._estimate_plan_event_times = py_types.MethodType(_estimate_plan_event_times, adapter)
        adapter._task_predicted_score_from_times = py_types.MethodType(_task_score, adapter)
        adapter._road_distance = py_types.MethodType(_road_distance, adapter)

        res_index = {
            "base": SimpleNamespace(id="base", state=RLControllerAdapter.STATE_REQUESTED),
            "t1": SimpleNamespace(id="t1", state=RLControllerAdapter.STATE_REQUESTED),
        }
        task = Task(
            id="t1",
            fromEdge="task_edge",
            toEdge="task_to",
            state=RLControllerAdapter.STATE_REQUESTED,
            reservationTime=0.0,
            estTravelTime=10.0,
            pickupDeadline=1000.0,
            dropoffDeadline=2000.0,
        )
        before_shadow = {k: list(v) for k, v in adapter._shadow_plan_by_robot.items()}
        _ = adapter.evaluate_marginal_insertion(
            robot_id="r0",
            task=task,
            res_index=res_index,
            current_edge="edge_r0",
            current_customer_ids=[],
            base_score_cache={},
        )
        self.assertEqual(before_shadow, adapter._shadow_plan_by_robot)

    def test_9_score_consistency_with_original_joint_function(self):
        adapter = object.__new__(RLControllerAdapter)
        adapter._shadow_plan_by_robot = {"r0": ["base"]}
        adapter._task_lifecycle = {}
        adapter.reward_weights = {"comp": 3.0, "wait": 1.5, "travel": 1.25}
        adapter.reward_caps = {"wait": 600.0, "travel": 240.0}

        def _is_completed(self, res_obj):
            _ = res_obj
            return False

        def _is_picked(self, res_obj):
            _ = res_obj
            return False

        def _task_from_reservation(self, res_id, res_obj):
            _ = res_obj
            return Task(
                id=str(res_id),
                fromEdge="x",
                toEdge="y",
                state=RLControllerAdapter.STATE_REQUESTED,
                reservationTime=0.0,
                estTravelTime=10.0,
                pickupDeadline=1000.0,
                dropoffDeadline=2000.0,
            )

        def _estimate_plan_event_times(self, *, rid, res_ids, res_index, current_edge, current_customer_ids):
            _ = (rid, res_index, current_edge, current_customer_ids)
            out = {}
            t = 5.0
            for res_id in res_ids:
                out[str(res_id)] = {"pickup": t, "dropoff": t + 10.0}
                t += 2.0
            return out

        def _task_score(self, *, task, predicted_pickup_time, predicted_dropoff_time, actual_pickup_time, preserve_actual_pickup_validity):
            _ = (task, actual_pickup_time, preserve_actual_pickup_validity)
            return float(predicted_dropoff_time - predicted_pickup_time)

        def _road_distance(self, from_edge, to_edge):
            _ = (from_edge, to_edge)
            return 10.0

        adapter._is_completed = py_types.MethodType(_is_completed, adapter)
        adapter._is_picked = py_types.MethodType(_is_picked, adapter)
        adapter._task_from_reservation = py_types.MethodType(_task_from_reservation, adapter)
        adapter._estimate_plan_event_times = py_types.MethodType(_estimate_plan_event_times, adapter)
        adapter._task_predicted_score_from_times = py_types.MethodType(_task_score, adapter)
        adapter._road_distance = py_types.MethodType(_road_distance, adapter)

        res_index = {
            "base": SimpleNamespace(id="base", state=RLControllerAdapter.STATE_REQUESTED),
            "t1": SimpleNamespace(id="t1", state=RLControllerAdapter.STATE_REQUESTED),
        }
        task = Task(
            id="t1",
            fromEdge="task_edge",
            toEdge="task_to",
            state=RLControllerAdapter.STATE_REQUESTED,
            reservationTime=0.0,
            estTravelTime=10.0,
            pickupDeadline=1000.0,
            dropoffDeadline=2000.0,
        )

        eval_new = adapter.evaluate_marginal_insertion(
            robot_id="r0",
            task=task,
            res_index=res_index,
            current_edge="edge_r0",
            current_customer_ids=[],
            base_score_cache={},
        )
        marginal_old, after_old, pickup_old, dist_old = adapter._estimate_joint_candidate_marginal_prediction(
            rid="r0",
            candidate=task,
            res_index=res_index,
            current_edge="edge_r0",
            current_customer_ids=[],
        )

        self.assertAlmostEqual(eval_new.marginal_score, marginal_old)
        self.assertAlmostEqual(eval_new.inserted_route_score, after_old)
        self.assertAlmostEqual(eval_new.candidate_pickup_time, pickup_old)
        self.assertAlmostEqual(eval_new.pickup_distance, dist_old)


if __name__ == "__main__":
    unittest.main()
