from __future__ import annotations
import numpy as np
from typing import Any, Optional, Tuple, List, cast
from sumo_rl_rs.environment.rl_controller_adapter import RLControllerAdapter, Task

BASE_ROBOT_FEATURE_NAMES: List[str] = [
    "taxi_loc_x", "taxi_loc_y", "taxi_current_capacity",
    "pad4", "pad5", "pad6", "pad7", "pad8", "pad9",
]
BASE_TASK_FEATURE_NAMES: List[str] = [
    "release_time_s", "waiting_time_s", "est_travel_time_s",
    "pickup_loc_x", "pickup_loc_y", "drop_loc_x", "drop_loc_y",
    "is_obsolete", "is_assigned",
]

def compute_feature_dim(
    use_xy_pickup: bool = False,
    use_node_type: bool = False,
    use_edge_rt: bool = False,
    use_ego_robot: bool = False,
    robot_commitment: str = "none",
    route_slots_k: int = 2,
) -> int:
    dim = 9
    if use_xy_pickup and not use_edge_rt:
        dim += 2
    if use_node_type:
        dim += 2
    if use_ego_robot:
        dim += 1
    return dim

def get_feature_names(
    use_xy_pickup: bool = False,
    use_node_type: bool = False,
    use_edge_rt: bool = False,
    use_ego_robot: bool = False,
    robot_commitment: str = "none",
    route_slots_k: int = 2,
) -> tuple[List[str], List[str]]:
    robot_names = list(BASE_ROBOT_FEATURE_NAMES)
    task_names = list(BASE_TASK_FEATURE_NAMES)

    if use_xy_pickup and not use_edge_rt:
        robot_names += ["pad10", "pad11"]
        task_names = [
            "release_time_s", "waiting_time_s", "est_travel_time_s",
            "pickup_loc_x", "pickup_loc_y", "pickup_dx", "pickup_dy",
            "drop_loc_x", "drop_loc_y", "is_obsolete", "is_assigned",
        ]

    if use_node_type:
        robot_names += ["is_robot", "is_task"]
        task_names += ["is_robot", "is_task"]

    if use_ego_robot:
        robot_names += ["is_ego_robot"]
        task_names += ["is_ego_robot"]

    return robot_names, task_names


def expand_edge_features(
    edge_features: Optional[List[str]],
    robot_commitment: str = "none",
    route_slots_k: int = 2,
) -> List[str]:
    feats = list(edge_features or [])
    if robot_commitment != "route_slots":
        return feats
    for idx in range(int(route_slots_k)):
        slot_names = [
            f"slot{idx}_pu_dx",
            f"slot{idx}_pu_dy",
            f"slot{idx}_do_dx",
            f"slot{idx}_do_dy",
            f"slot{idx}_valid",
        ]
        for name in slot_names:
            if name not in feats:
                feats.append(name)
    return feats

def make_feature_fn(
    ctrl: RLControllerAdapter,
    use_xy_pickup: bool = False,
    normalize_features: bool = False,
    use_node_type: bool = False,
    use_edge_rt: bool = False,
    edge_features: Optional[List[str]] = None,
    use_ego_robot: bool = False,
    robot_commitment: str = "none",
    route_slots_k: int = 2,
):
    feature_dim = compute_feature_dim(
        use_xy_pickup=use_xy_pickup,
        use_node_type=use_node_type,
        use_edge_rt=use_edge_rt,
        use_ego_robot=use_ego_robot,
        robot_commitment=robot_commitment,
        route_slots_k=route_slots_k,
    )
    edge_features = expand_edge_features(edge_features, robot_commitment, route_slots_k)
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

    def _append_node_type(out: np.ndarray, node_type: str) -> None:
        if not use_node_type:
            return
        if use_ego_robot:
            if out.shape[0] >= 3:
                out[-3] = 1.0 if node_type == "robot" else 0.0
                out[-2] = 1.0 if node_type == "task" else 0.0
        else:
            if out.shape[0] >= 2:
                out[-2] = 1.0 if node_type == "robot" else 0.0
                out[-1] = 1.0 if node_type == "task" else 0.0

    def _append_ego_robot(out: np.ndarray, is_ego: bool) -> None:
        if not use_ego_robot:
            return
        if out.shape[0] >= 1:
            out[-1] = 1.0 if is_ego else 0.0

    def _robot_route_slots(rid_s: str) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        if robot_commitment != "route_slots":
            return []
        res_index = ctrl._reservation_index()
        seq = list(ctrl._shadow_plan_by_robot.get(rid_s, []))
        if not seq:
            return []
        seen: set[str] = set()
        out_slots: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
        for res_id in seq:
            res_obj = res_index.get(str(res_id))
            if res_obj is None:
                continue
            res_key = str(res_id)
            if res_key in seen:
                continue
            seen.add(res_key)

            pu_edge = ctrl._edge_for_pickup(res_obj)
            do_edge = ctrl._edge_for_dropoff(res_obj)
            pu_xy = _edge_xy(pu_edge)
            do_xy = _edge_xy(do_edge)
            out_slots.append((pu_xy, do_xy, 1.0))
            if len(out_slots) >= int(route_slots_k):
                break
        return out_slots

    def _edge_rt_features(rid_s: str, t: Task) -> np.ndarray:
        out = np.zeros((len(edge_features),), dtype=np.float32)
        if not edge_features:
            return out
        rx, ry = _robot_xy(rid_s)
        px, py = _edge_xy(getattr(t, "fromEdge", None))
        tx_do, ty_do = _edge_xy(getattr(t, "toEdge", None))
        dx = float(px - rx)
        dy = float(py - ry)
        if normalize_features:
            dx /= pos_scale
            dy /= pos_scale
        eta = 0.0
        try:
            route = ctrl.sumo.vehicle.getRoute(rid_s)
            r_edge = route[0] if route else ""
            dist = float(ctrl._road_distance(r_edge, getattr(t, "fromEdge", "")))
            eta = dist / 10.0 if np.isfinite(dist) else 0.0
        except Exception:
            eta = 0.0
        if normalize_features:
            eta = float(np.clip(eta / travel_scale, 0.0, 1.0))

        slot_values: dict[str, float] = {}
        if robot_commitment == "route_slots":
            slots = _robot_route_slots(rid_s)
            for s_idx in range(int(route_slots_k)):
                if s_idx < len(slots):
                    (pu_xy, do_xy, valid) = slots[s_idx]
                    pu_dx = float(pu_xy[0] - px)
                    pu_dy = float(pu_xy[1] - py)
                    do_dx = float(do_xy[0] - tx_do)
                    do_dy = float(do_xy[1] - ty_do)
                    if normalize_features:
                        pu_dx /= pos_scale
                        pu_dy /= pos_scale
                        do_dx /= pos_scale
                        do_dy /= pos_scale
                    slot_values[f"slot{s_idx}_pu_dx"] = pu_dx
                    slot_values[f"slot{s_idx}_pu_dy"] = pu_dy
                    slot_values[f"slot{s_idx}_do_dx"] = do_dx
                    slot_values[f"slot{s_idx}_do_dy"] = do_dy
                    slot_values[f"slot{s_idx}_valid"] = float(valid)
                else:
                    slot_values[f"slot{s_idx}_pu_dx"] = 0.0
                    slot_values[f"slot{s_idx}_pu_dy"] = 0.0
                    slot_values[f"slot{s_idx}_do_dx"] = 0.0
                    slot_values[f"slot{s_idx}_do_dy"] = 0.0
                    slot_values[f"slot{s_idx}_valid"] = 0.0

        for i, name in enumerate(edge_features):
            if name == "dx":
                out[i] = dx
            elif name == "dy":
                out[i] = dy
            elif name == "eta":
                out[i] = eta
            elif name == "is_ego_edge":
                out[i] = 0.0
            elif name in slot_values:
                out[i] = slot_values[name]
        return out

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
        out = np.zeros((feature_dim,), dtype=np.float32)
        now = ctrl._now()

        # robot features are [ robot_x_coordinate, robot_y_coordinate, free_capacity, 0, 0, 0, 0, 0, 0 ]
        # example: [2.0107662e+03 2.4869739e+03 2.0000000e+00 0. 0. 0. 0. 0. 0.]

        if node_type in {"robot", "robot_ego", "robot_other"}:
            is_ego = node_type != "robot_other"
            rid = _normalize_rid(obj_a)
            if not _valid_robot_id(rid):
                return out  # all zeros for padded/missing taxis

            rid_s = cast(str, rid)

            rx, ry = _robot_xy(rid_s)
            if normalize_features:
                out[0], out[1] = rx / pos_scale, ry / pos_scale
            else:
                out[0], out[1] = rx, ry

            try:
                cap = max(1, ctrl._get_vehicle_capacity(rid_s))
            except Exception:
                cap = max(1, ctrl.max_robot_capacity)
            try:
                onboard = len(ctrl._get_current_customers(rid_s))
            except Exception:
                onboard = 0
            if normalize_features:
                out[2] = float(max(0, cap - onboard)) / cap_scale
            else:
                out[2] = float(max(0, cap - onboard))
            _append_node_type(out, "robot")
            _append_ego_robot(out, is_ego)
            return out

        # task features are:
        # [
        #     reservationTime,
        #     waitingTime,
        #     estTravelTime,
        #     pickup_x, pickup_y,
        #     drop_x, drop_y,
        #     is_obsolete,
        #     is_assigned
        # ]
        elif node_type == "task":
            _ = _normalize_rid(obj_a)
            t = _resolve_task(obj_b)
            if t is None:
                return out

            if normalize_features:
                out[0] = float(t.reservationTime) / time_scale
                out[1] = float(max(0.0, now - float(t.reservationTime))) / wait_scale
                out[2] = float(getattr(t, "estTravelTime", 0.0)) / travel_scale
            else:
                out[0] = float(t.reservationTime)
                out[1] = float(max(0.0, now - float(t.reservationTime)))
                out[2] = float(getattr(t, "estTravelTime", 0.0))

            # fromEdge = getattr(t, "fromEdge", None)
            # toEdge = getattr(t, "toEdge", None)

            # print("fromEdge:", fromEdge)
            # print("toEdge: ", toEdge)

            px, py = _edge_xy(getattr(t, "fromEdge", None))
            dx, dy = _edge_xy(getattr(t, "toEdge", None))
            if normalize_features:
                out[3], out[4] = px / pos_scale, py / pos_scale
            else:
                out[3], out[4] = px, py
            if use_xy_pickup and not use_edge_rt:
                rx, ry = _robot_xy(_normalize_rid(obj_a))
                if normalize_features:
                    out[5] = float(px - rx) / pos_scale
                    out[6] = float(py - ry) / pos_scale
                    out[7], out[8] = dx / pos_scale, dy / pos_scale
                else:
                    out[5] = float(px - rx)
                    out[6] = float(py - ry)
                    out[7], out[8] = dx, dy
                out[9] = 1.0 if bool(getattr(t, "is_obsolete", False)) else 0.0
                out[10] = 1.0 if bool(getattr(t, "is_assigned", False)) else 0.0
            else:
                if normalize_features:
                    out[5], out[6] = dx / pos_scale, dy / pos_scale
                else:
                    out[5], out[6] = dx, dy
                out[7] = 1.0 if bool(getattr(t, "is_obsolete", False)) else 0.0
                out[8] = 1.0 if bool(getattr(t, "is_assigned", False)) else 0.0
            _append_node_type(out, "task")
            _append_ego_robot(out, False)
            return out

        elif node_type == "edge_rt":
            rid = _normalize_rid(obj_a)
            t = _resolve_task(obj_b)
            if rid is None or t is None:
                return np.zeros((len(edge_features),), dtype=np.float32)
            return _edge_rt_features(cast(str, rid), t)

        return out

    return feature_fn
