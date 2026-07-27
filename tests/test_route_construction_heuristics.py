import types
import unittest
from types import SimpleNamespace
from pathlib import Path
import importlib.util
import sys
import math
import types as py_types


try:
    import numpy  # type: ignore  # noqa: F401
except Exception:
    # Minimal stub for controller import in lightweight unit-test environments.
    np_stub = SimpleNamespace(
        isfinite=lambda x: math.isfinite(float(x)),
    )
    sys.modules["numpy"] = np_stub


if "sumo_rl_rs" not in sys.modules:
    pkg = py_types.ModuleType("sumo_rl_rs")
    pkg.__path__ = []  # mark as package
    sys.modules["sumo_rl_rs"] = pkg

if "sumo_rl_rs.logging" not in sys.modules:
    logging_pkg = py_types.ModuleType("sumo_rl_rs.logging")
    logging_pkg.__path__ = []
    sys.modules["sumo_rl_rs.logging"] = logging_pkg

if "sumo_rl_rs.logging.ridepool_logger" not in sys.modules:
    logger_mod = py_types.ModuleType("sumo_rl_rs.logging.ridepool_logger")

    class _RidepoolLogger:  # pragma: no cover - test stub
        pass

    logger_mod.RidepoolLogger = _RidepoolLogger
    sys.modules["sumo_rl_rs.logging.ridepool_logger"] = logger_mod


_CONTROLLER_PATH = Path(__file__).resolve().parents[1] / "sumo_rl_rs" / "environment" / "rl_controller_adapter.py"
_SPEC = importlib.util.spec_from_file_location("route_construction_controller", _CONTROLLER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load controller module from {_CONTROLLER_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

RLControllerAdapter = _MODULE.RLControllerAdapter
Task = _MODULE.Task
RouteScore = _MODULE.RouteScore
RouteStop = _MODULE.RouteStop


class RouteConstructionHeuristicTests(unittest.TestCase):
    def _fake_adapter(self, travel_times, *, mode="deadline_travel", max_stops=8, capacity=2):
        adapter = object.__new__(RLControllerAdapter)
        adapter.route_construction = mode
        adapter.route_exhaustive_max_stops = int(max_stops)
        adapter.route_construction_debug = False
        adapter._last_route_construction_diag_by_robot = {}
        adapter.logger = None
        adapter.completion_mode = "valid_dropoff"
        adapter.reward_type = "wait_travel"
        adapter.reward_weights = {
            "comp": 3.0,
            "wait": 1.5,
            "deadline": 10.0,
            "travel": 1.25,
        }
        adapter.reward_caps = {
            "wait": 600.0,
            "deadline": 600.0,
            "travel": 240.0,
        }
        adapter.max_wait_delay_s = 1.0
        adapter.max_travel_delay_s = 0.0
        adapter._task_lifecycle = {}
        adapter.STATE_REQUESTED = RLControllerAdapter.STATE_REQUESTED
        adapter.STATE_COMPLETED = RLControllerAdapter.STATE_COMPLETED

        def _estimate_travel_time(self, from_edge, to_edge):
            return float(travel_times.get((str(from_edge), str(to_edge)), 1e6))

        def _get_vehicle_capacity(self, rid):
            return int(capacity)

        def _greedy_pd_sequence(self, rid, res_ids, res_index, already_picked):
            # Stable fallback stub: pickup then dropoff order by base order.
            seq = []
            for res_id in res_ids:
                if res_id in already_picked:
                    seq.extend([res_id, res_id])
                else:
                    seq.extend([res_id, res_id])
            return seq

        adapter._estimate_travel_time = types.MethodType(_estimate_travel_time, adapter)
        adapter._get_vehicle_capacity = types.MethodType(_get_vehicle_capacity, adapter)
        adapter._greedy_pd_sequence = types.MethodType(_greedy_pd_sequence, adapter)
        return adapter

    def _res(self, rid, from_edge, to_edge, reservation_time=0.0, state=1):
        return SimpleNamespace(
            id=str(rid),
            fromEdge=str(from_edge),
            toEdge=str(to_edge),
            reservationTime=float(reservation_time),
            state=int(state),
        )

    def test_1_precedence(self):
        stops = [
            RouteStop("A", "pickup", "A_pu"),
            RouteStop("A", "dropoff", "A_do"),
            RouteStop("B", "pickup", "B_pu"),
            RouteStop("B", "dropoff", "B_do"),
        ]
        seqs = RLControllerAdapter._generate_valid_stop_orders(
            stops=stops,
            initially_picked=set(),
            initial_occupancy=0,
            capacity=2,
        )
        self.assertGreater(len(seqs), 0)
        for seq in seqs:
            labels = [(s.res_id, s.kind) for s in seq]
            self.assertLess(labels.index(("A", "pickup")), labels.index(("A", "dropoff")))
            self.assertLess(labels.index(("B", "pickup")), labels.index(("B", "dropoff")))

    def test_2_capacity(self):
        stops = [
            RouteStop("A", "pickup", "A_pu"),
            RouteStop("A", "dropoff", "A_do"),
            RouteStop("B", "pickup", "B_pu"),
            RouteStop("B", "dropoff", "B_do"),
        ]
        seqs = RLControllerAdapter._generate_valid_stop_orders(
            stops=stops,
            initially_picked=set(),
            initial_occupancy=0,
            capacity=1,
        )
        self.assertGreater(len(seqs), 0)
        for seq in seqs:
            occ = 0
            for stop in seq:
                occ += 1 if stop.kind == "pickup" else -1
                self.assertLessEqual(occ, 1)
                self.assertGreaterEqual(occ, 0)

    def test_3_onboard_passenger(self):
        # Picked-up passenger contributes only dropoff and no additional wait penalty.
        adapter = self._fake_adapter({}, capacity=2)
        sequence = [RouteStop("P", "dropoff", "P_do")]
        score, _ = adapter._score_route_stop_order(
            sequence=sequence,
            start_edge="S",
            start_time=0.0,
            travel_time_fn=lambda a, b: 1.0,
            tasks_by_res={
                "P": Task(
                    id="P",
                    fromEdge="P_from",
                    toEdge="P_do",
                    state=RLControllerAdapter.STATE_PICKED_UP,
                    reservationTime=0.0,
                    estTravelTime=1.0,
                    pickupDeadline=-100.0,
                    dropoffDeadline=10.0,
                )
            },
            actual_pickup_time_by_res={"P": 0.0},
            initially_picked={"P"},
        )
        self.assertEqual(score.total_weighted_wait_penalty, 0.0)

    def test_4_deadline_priority(self):
        travel = {
            ("P_from", "P_to"): 1.0,
            ("U_from", "U_to"): 1.0,
            ("S", "P_to"): 2.0,
            ("S", "U_from"): 1.0,
            ("U_from", "U_to"): 1.0,
            ("U_to", "P_to"): 1.0,
            ("P_to", "U_from"): 3.0,
        }
        adapter = self._fake_adapter(travel, capacity=2)
        res_index = {
            "P": self._res("P", "P_from", "P_to", reservation_time=0.0, state=RLControllerAdapter.STATE_PICKED_UP),
            "U": self._res("U", "U_from", "U_to", reservation_time=10.0, state=RLControllerAdapter.STATE_REQUESTED),
        }
        seq = adapter._build_pd_sequence(
            rid="r0",
            base_ids=["P", "U"],
            res_index=res_index,
            already_picked={"P"},
            current_edge="S",
            start_time=0.0,
        )
        # Prefix includes synthetic pickup for already picked P; second P is actual dropoff.
        self.assertEqual(seq, ["P", "P", "U", "U"])

    def test_5_lateness_tie_breaker(self):
        travel = {
            ("P_from", "P_to"): 1.0,
            ("U_from", "U_to"): 1.0,
            ("S", "P_to"): 3.0,
            ("S", "U_from"): 1.0,
            ("U_from", "U_to"): 1.0,
            ("U_to", "P_to"): 2.0,
            ("P_to", "U_from"): 1.0,
        }
        adapter = self._fake_adapter(travel, capacity=2)
        res_index = {
            "P": self._res("P", "P_from", "P_to", reservation_time=0.0, state=RLControllerAdapter.STATE_PICKED_UP),
            "U": self._res("U", "U_from", "U_to", reservation_time=10.0, state=RLControllerAdapter.STATE_REQUESTED),
        }
        seq = adapter._build_pd_sequence(
            rid="r0",
            base_ids=["P", "U"],
            res_index=res_index,
            already_picked={"P"},
            current_edge="S",
            start_time=0.0,
        )
        self.assertEqual(seq, ["P", "P", "U", "U"])

    def test_6_extreme_passenger_detour_priority(self):
        a = RouteScore(1.0, 5.0, 6.0, 10.0, 1, -0.1, -0.1)
        b = RouteScore(1.0, 4.0, 7.0, 10.0, 1, -0.1, -0.1)
        sig = (("A", "pickup"),)
        self.assertLess(RLControllerAdapter._score_sort_key(b, sig), RLControllerAdapter._score_sort_key(a, sig))

    def test_7_route_duration_tie_breaker(self):
        a = RouteScore(1.0, 1.0, 2.0, 12.0, 1, -0.1, -0.1)
        b = RouteScore(1.0, 1.0, 2.0, 11.0, 1, -0.1, -0.1)
        sig = (("A", "pickup"),)
        self.assertLess(RLControllerAdapter._score_sort_key(b, sig), RLControllerAdapter._score_sort_key(a, sig))

    def test_8_deterministic_equality(self):
        travel = {
            ("A_from", "A_to"): 1.0,
            ("B_from", "B_to"): 1.0,
            ("S", "A_from"): 1.0,
            ("S", "B_from"): 1.0,
            ("A_to", "B_from"): 1.0,
            ("B_to", "A_from"): 1.0,
        }
        adapter = self._fake_adapter(travel, capacity=1)
        res_index = {
            "A": self._res("A", "A_from", "A_to", reservation_time=100.0, state=RLControllerAdapter.STATE_REQUESTED),
            "B": self._res("B", "B_from", "B_to", reservation_time=100.0, state=RLControllerAdapter.STATE_REQUESTED),
        }

        out = []
        for _ in range(5):
            out.append(
                adapter._build_pd_sequence(
                    rid="r0",
                    base_ids=["A", "B"],
                    res_index=res_index,
                    already_picked=set(),
                    current_edge="S",
                    start_time=0.0,
                )
            )

        self.assertTrue(all(seq == out[0] for seq in out))
        self.assertEqual(out[0], ["A", "A", "B", "B"])

    def test_9_fallback(self):
        travel = {
            ("A_from", "A_to"): 1.0,
            ("B_from", "B_to"): 1.0,
            ("S", "A_from"): 1.0,
            ("S", "B_from"): 1.0,
            ("A_to", "B_from"): 1.0,
            ("B_to", "A_from"): 1.0,
        }
        adapter = self._fake_adapter(travel, max_stops=2, capacity=2)

        def _fallback_only(self, rid, res_ids, res_index, already_picked):
            return ["FALLBACK_USED"]

        adapter._greedy_pd_sequence = types.MethodType(_fallback_only, adapter)
        res_index = {
            "A": self._res("A", "A_from", "A_to", reservation_time=0.0, state=RLControllerAdapter.STATE_REQUESTED),
            "B": self._res("B", "B_from", "B_to", reservation_time=0.0, state=RLControllerAdapter.STATE_REQUESTED),
        }
        seq = adapter._build_pd_sequence(
            rid="r0",
            base_ids=["A", "B"],
            res_index=res_index,
            already_picked=set(),
            current_edge="S",
            start_time=0.0,
        )
        self.assertEqual(seq, ["FALLBACK_USED"])
        diag = adapter._last_route_construction_diag_by_robot.get("r0", {})
        self.assertTrue(bool(diag.get("fallback", False)))

    def test_10_no_remaining_stops(self):
        adapter = self._fake_adapter({}, max_stops=8, capacity=2)
        seq = adapter._build_pd_sequence(
            rid="r0",
            base_ids=[],
            res_index={},
            already_picked=set(),
            current_edge="S",
            start_time=0.0,
        )
        self.assertEqual(seq, [])


if __name__ == "__main__":
    unittest.main()
