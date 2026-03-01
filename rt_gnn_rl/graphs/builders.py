"""
Graph construction utilities for batched ego-graphs.

build_padded_ego_batch(...) converts per-robot candidate lists into a fixed-shape
batch of ego-graphs (robot + candidate tasks) suitable for GNN-based policies.
"""
from __future__ import annotations
from typing import Callable, Sequence, Any, List, Tuple, Optional, Dict
import numpy as np

# Only the fields required by feature_fn are accessed; all other fields are passed through unchanged.
def build_padded_ego_batch(
    *,
    robots: Sequence[Optional[str]],
    tasks: Sequence[Any],
    candidate_lists: Sequence[Sequence[int]],  # per-robot, indices into `tasks`
    N_max: int,
    E_max: int,
    K_max: int,
    F: int,
    G: int,
    feature_fn: Callable[[Any, Any, str], np.ndarray],
    two_hop: bool = False,
    normalize_features: bool = False,
    vicinity_m: float = 0.0,
    # global_stats_fn: Optional[Callable[[Any], np.ndarray]] = None,
) -> Tuple[Dict[str, np.ndarray], List[List[Optional[str]]]]:
    """
    Build per-robot ego-graphs as star graphs:
      node 0 = robot, nodes 1..M = that robot's candidate tasks (truncated to fit N_max).
      Edges: undirected star (robot<->task).

    Parameters
    ----------
    robots : list
        List of robot (or vehicle) state objects. Each element contains the
        attributes of one robot (e.g. position, capacity, load).
        Length = number of robots R.

    tasks : Sequence[Any]
        Sequence mapping task_id -> task state object. Each task state includes
        attributes like location, release time, deadline, etc.

    candidate_lists : list[list[int]]
        For each robot i, a list of task_ids that are feasible candidates at this step
        (e.g. nearby requests, reachable within detour limit).
        Length = R. Inner list lengths vary (0 .. K_max).

    N_max : int
        Maximum number of nodes allowed in any ego-graph (robot + tasks).
        Used to pad node features and masks to a fixed size.

    E_max : int
        Maximum number of edges allowed in any ego-graph.
        Used to pad edge indices and masks to a fixed size.

    K_max : int
        Maximum number of candidate tasks considered per robot.
        Defines the action space size; candidate lists are padded/truncated to K_max.

    F: int
        Node feature dim.

    G: int
        Global stats dim.

    feature_fn : callable
        Function that converts (robot_state, task_state, node_type) into a feature
        vector (numpy array of length F) for a node.
        Special case: when called with "probe_dim", should return the feature length F.

    global_stats_fn : callable
        Function that returns a global feature vector (length G) describing the system
        state (e.g. current time, number of tasks waiting).
        Called once per environment step.

    Returns:
      obs dict and a parallel list cand_task_ids[R][K] mapping each slot to a task_id (or None if padded).
    """
    R = len(robots)
    x = np.zeros((R, N_max, F), np.float32)
    node_mask = np.zeros((R, N_max), np.uint8)
    edge_index = np.zeros((R, 2, E_max), np.int64)
    edge_mask = np.zeros((R, E_max), np.uint8)
    cand_idx = np.zeros((R, K_max), np.int64)
    cand_mask = np.zeros((R, K_max), np.uint8)

    # map slot -> task_id used this step (to feed controller in step())
    cand_task_ids: List[List[Optional[str]]] = [[None] * K_max for _ in range(R)]

    vicinity_threshold = float(vicinity_m)
    pos_scale = max(1.0, float(vicinity_m))

    def _to_meters(xy: Tuple[float, float]) -> Tuple[float, float]:
        if normalize_features:
            return xy[0] * pos_scale, xy[1] * pos_scale
        return xy

    robot_xy_cache: List[Tuple[float, float]] = []
    for rid in robots:
        try:
            rf = feature_fn(rid, None, "robot")
            robot_xy_cache.append(_to_meters((float(rf[0]), float(rf[1]))))
        except Exception:
            robot_xy_cache.append((0.0, 0.0))

    for i in range(R):
        rid = robots[i]
        # Robot node at index 0 (even if rid is None, create a dummy vector)
        try:
            x[i, 0, :] = feature_fn(rid, None, "robot")
        except Exception:
            # fallback: zeros with bias 1 in the last slot
            pass
        node_mask[i, 0] = 1

        # Fill task nodes from candidates
        cands = list(candidate_lists[i]) if i < len(candidate_lists) else []
        # Cap by available node slots (leave index 0 for robot)
        max_tasks_here = max(0, N_max - 1)
        cands = cands[: min(K_max, max_tasks_here)]

        # Add nodes and edges
        e_ptr = 0
        next_node_id = 1 + len(cands)
        competitor_nodes: Dict[str, int] = {}
        for local_slot, task_idx in enumerate(cands):
            node_id = 1 + local_slot  # node indices for tasks
            if node_id >= N_max:
                break

            t = tasks[task_idx]
            # Task features conditioned on robot context -> pass rid as obj_a, t as obj_b
            try:
                x[i, node_id, :] = feature_fn(rid, t, "task")
            except Exception:
                pass
            node_mask[i, node_id] = 1

            task_xy = None
            if two_hop:
                try:
                    tf = feature_fn(rid, t, "task")
                    task_xy = _to_meters((float(tf[3]), float(tf[4])))
                except Exception:
                    task_xy = None

            # cand slot mapping: slot `local_slot` → node index `node_id`
            cand_idx[i, local_slot] = node_id
            cand_mask[i, local_slot] = 1

            # Store external task identifiers for downstream controller mapping.
            try:
                cand_task_ids[i][local_slot] = str(getattr(t, "id", None)) or None
            except Exception:
                cand_task_ids[i][local_slot] = None

            # add undirected star edges robot<->task (two directed edges)
            if e_ptr + 2 <= E_max:
                edge_index[i, 0, e_ptr] = 0
                edge_index[i, 1, e_ptr] = node_id
                edge_mask[i, e_ptr] = 1
                e_ptr += 1

                edge_index[i, 0, e_ptr] = node_id
                edge_index[i, 1, e_ptr] = 0
                edge_mask[i, e_ptr] = 1
                e_ptr += 1

            if two_hop and task_xy is not None:
                for j, other_rid in enumerate(robots):
                    if other_rid is None or j == i:
                        continue
                    rx, ry = robot_xy_cache[j]
                    dx = rx - task_xy[0]
                    dy = ry - task_xy[1]
                    if (dx * dx + dy * dy) > (vicinity_threshold * vicinity_threshold):
                        continue

                    other_key = str(other_rid)
                    if other_key in competitor_nodes:
                        other_node_id = competitor_nodes[other_key]
                    else:
                        if next_node_id >= N_max:
                            continue
                        other_node_id = next_node_id
                        next_node_id += 1
                        competitor_nodes[other_key] = other_node_id
                        try:
                            x[i, other_node_id, :] = feature_fn(other_rid, None, "robot")
                        except Exception:
                            pass
                        node_mask[i, other_node_id] = 1

                    if e_ptr + 2 <= E_max:
                        edge_index[i, 0, e_ptr] = node_id
                        edge_index[i, 1, e_ptr] = other_node_id
                        edge_mask[i, e_ptr] = 1
                        e_ptr += 1

                        edge_index[i, 0, e_ptr] = other_node_id
                        edge_index[i, 1, e_ptr] = node_id
                        edge_mask[i, e_ptr] = 1
                        e_ptr += 1
        # any remaining cand slots are already 0/False (padding)

    obs = dict(
        x=x,
        node_mask=node_mask,
        edge_index=edge_index,
        edge_mask=edge_mask,
        cand_idx=cand_idx,
        cand_mask=cand_mask,
        # global_stats=(global_stats_fn() if global_stats_fn else np.zeros((G,), np.float32)),
    )
    return obs, cand_task_ids
