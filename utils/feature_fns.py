from __future__ import annotations
import numpy as np
from typing import Any, Optional, Tuple, List, cast
from sumo_rl_rs.environment.rl_controller_adapter import RLControllerAdapter, Task

ROBOT_FEATURE_NAMES: List[str] = [
    "taxi_loc_x", "taxi_loc_y", "taxi_current_capacity",
    "pad4", "pad5", "pad6", "pad7", "pad8", "pad9", "pad10", "pad11",
]
TASK_FEATURE_NAMES: List[str] = [
    "release_time_s", "waiting_time_s", "est_travel_time_s",
    "pickup_loc_x", "pickup_loc_y", "pickup_dx", "pickup_dy",
    "drop_loc_x", "drop_loc_y", "is_obsolete", "is_assigned",
]
# robot node and task node should have the same dimensionality, so robot features are padded with zeros
F = 11

def make_feature_fn(ctrl: RLControllerAdapter):
    # ---- static scaling -------------------------------------------------
    pos_scale = max(1.0, float(getattr(ctrl, "vicinity_m", 1000.0)))
    cap_scale = max(1.0, float(getattr(ctrl, "max_robot_capacity", 1)))
    wait_scale = max(1.0, float(getattr(ctrl, "max_wait_delay_s", 1.0)))
    travel_scale = max(1.0, float(getattr(ctrl, "max_travel_delay_s", 1.0)))
    if getattr(ctrl, "max_steps", None):
        time_scale = max(1.0, float(getattr(ctrl, "max_steps", 1.0)))
    else:
        time_scale = wait_scale

    def _normalize_rid(x: Any) -> Optional[str]:
        # Accept only real str ids; filter out None and the literal string "None"
        if isinstance(x, str) and x and x.lower() != "none":
            return x
        return None

    def _fleet_ids() -> set[str]:
        try:
            return set(ctrl.sumo.vehicle.getIDList())
        except Exception:
            return set()
        
    # ---- helpers accept Optional[str] ---------------------------------
    def _valid_robot_id(rid: Optional[str]) -> bool:
        return rid is not None and rid in _fleet_ids()

    def _robot_xy(rid: Optional[str]) -> Tuple[float, float]:
        if not _valid_robot_id(rid):
            return 0.0, 0.0
        rid_s = cast(str, rid)
        try:
            x, y = ctrl.sumo.vehicle.getPosition(rid_s)
            return float(x), float(y)
        except Exception:
            try:
                rt = ctrl.sumo.vehicle.getRoute(rid_s)
                e0 = rt[0] if rt else None
                if e0:
                    shape = ctrl.sumo.edge.getShape(e0)
                    if shape:
                        x, y = shape[0]
                        return float(x), float(y)
            except Exception:
                pass
        return 0.0, 0.0

    def _edge_xy(edge_id: Optional[str]) -> Tuple[float, float]:
        if not edge_id:
            return 0.0, 0.0
        try:
            x, y = ctrl.sumo.simulation.convert2D(edge_id, 0.0)
            return float(x), float(y)
        except Exception:
            pass
        return 0.0, 0.0

    def _resolve_task(x: Any) -> Optional[Task]:
        if isinstance(x, Task):
            return x
        tid = str(x) if x is not None else ""
        for t in (ctrl._last_tasks or ctrl.get_tasks()):
            if t.id == tid:
                return t
        return None

    # ---- main fn -------------------------------------------------------
    def feature_fn(obj_a, obj_b, node_type: str) -> np.ndarray:
        out = np.zeros((F,), dtype=np.float32)
        now = ctrl._now()

        # robot features are [ robot_x_coordinate, robot_y_coordinate, free_capacity, 0, 0, 0, 0, 0, 0, 0, 0 ]
        # example: [2.0107662e+03 2.4869739e+03 2.0000000e+00 0. 0. 0. 0. 0. 0. 0. 0.]

        if node_type == "robot":
            rid = _normalize_rid(obj_a)
            if not _valid_robot_id(rid):
                return out  # all zeros for padded/missing taxis

            rid_s = cast(str, rid)

            rx, ry = _robot_xy(rid_s)         
            out[0], out[1] = rx / pos_scale, ry / pos_scale

            try:
                cap = max(1, ctrl._get_vehicle_capacity(rid_s))
            except Exception:
                cap = max(1, ctrl.max_robot_capacity)
            try:
                onboard = len(ctrl._get_current_customers(rid_s))
            except Exception:
                onboard = 0
            out[2] = float(max(0, cap - onboard)) / cap_scale
            return out

        # task features are:
        # [
        #     reservationTime,
        #     waitingTime,
        #     estTravelTime,
        #     pickup_x, pickup_y,
        #     pickup_dx, pickup_dy,
        #     drop_x, drop_y,
        #     is_obsolete,
        #     is_assigned
        # ]
        elif node_type == "task":
            _ = _normalize_rid(obj_a)
            t = _resolve_task(obj_b)
            if t is None:
                return out

            out[0] = float(t.reservationTime) / time_scale
            out[1] = float(max(0.0, now - float(t.reservationTime))) / wait_scale
            out[2] = float(getattr(t, "estTravelTime", 0.0)) / travel_scale

            # fromEdge = getattr(t, "fromEdge", None)
            # toEdge = getattr(t, "toEdge", None)

            # print("fromEdge:", fromEdge)
            # print("toEdge: ", toEdge)

            px, py = _edge_xy(getattr(t, "fromEdge", None))
            dx, dy = _edge_xy(getattr(t, "toEdge", None))
            out[3], out[4] = px / pos_scale, py / pos_scale

            rx, ry = _robot_xy(_normalize_rid(obj_a))
            out[5] = float(px - rx) / pos_scale
            out[6] = float(py - ry) / pos_scale

            out[7], out[8] = dx / pos_scale, dy / pos_scale

            out[9] = 1.0 if bool(getattr(t, "is_obsolete", False)) else 0.0
            out[10] = 1.0 if bool(getattr(t, "is_assigned", False)) else 0.0

            return out

        return out

    return feature_fn
