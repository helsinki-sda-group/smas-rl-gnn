# src/sumo_rl_rs/environment/rl_controller_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Set, Callable

import numpy as np
from sumo_rl_rs.logging.ridepool_logger import RidepoolLogger  

@dataclass
class Task:
    id: str
    fromEdge: str
    toEdge: str
    state: int
    reservationTime: float
    estTravelTime: float = 0.0           # sec, pickup→dropoff
    pickupDeadline: float = 0.0          # sec, release + max_wait_delay_s
    dropoffDeadline: float = 0.0         # sec, release + estTravelTime + max_travel_delay_s
    is_obsolete: bool = False            # missed pickup deadline
    is_assigned: bool = False            # reservation currently assigned to some taxi



class RLControllerAdapter:
    """
    RL adapter over TraCI's Taxi interface.

    Key features:
      • Candidate selection is restricted by a "vicinity" threshold (meters) using road-network distance.
      • Rewards are computed per-robot as in MultiTaskAllocationEnv:
          r_i = capacity_i + (-1 step) + (- abandoned_i) + (- wait_at_pickups_i) + completion_i
        where completion_i = pickups_i or dropoffs_i per tick (configurable).
    """

    # Reservation state flags 
    STATE_REQUESTED = 1
    STATE_ASSIGNED = 4
    STATE_PICKED_UP = 8
    STATE_COMPLETED = 16

    def __init__(
        self,
        sumo: Any,
        *,
        k_max: int = 8,
        vicinity_m: float = 2_000.0,
        max_steps: Optional[int] = None,
        min_episode_steps: int = 0,            # warmup; don't allow done before this many steps
        idle_patience_steps:int = 600,       # how long whole-fleet idle must persist
        respect_sumo_end:bool = False,      # stop at SUMO <time end="...">
        serve_to_empty: bool = True,           # end only when nothing left to do
        require_seen_reservation: bool = True, # don't allow done until we've seen at least one reservation
        completion_mode: str = "dropoff",
        reset_fn: Optional[Callable[[], None]] = None,
        max_wait_delay_s: float = 600.0,        # allowed waiting time until pickup
        max_travel_delay_s: float = 900.0,      # how late the robot is allowed to deliever to dropoff
        max_robot_capacity: int = 5,
        logger: Optional["RidepoolLogger"]=None,
    ) -> None:
        """
        Args:
            sumo: the TraCI handle/module (usually `traci`).
            k_max: cap on candidates per robot.
            vicinity_m: maximum road-network distance (meters) robot→task pickup to be a valid candidate.
            max_steps: hard episode max length.
            min_episode_steps: minimum episode length.
            idle_patience_steps: end if whole fleet is idle for that long
            respect_sumo_end: stop at SUMO <time end="...">
            serve_to_empty: to end the episode, there should be no pending reservations, active assignments or passengers travelling
            require_seen_reservation: episode cannot be marked as done unless at least one reservation was seen
            completion_mode: "pickup" → reward counted at pickup, "dropoff" → reward counted at dropoff (default).
            reset_fn: optional callable to (re)start/reset SUMO when env.reset() calls adapter.reset().
            max_wait_delay_s: allowed waiting time until pickup. If this is violated, the task becomes obsolete.
            max_travel_delay_s: how late the robot is allowed to deliever to dropoff. No explicit penalty for that in the current implementation. 
            max_robot_capacity:
            logger: RidepoolLogger instance.
        """
        self.sumo = sumo
        self.k_max = int(k_max)
        self.vicinity_m = float(vicinity_m)
        self.max_steps = max_steps
        self.completion_mode = completion_mode.lower().strip()
        assert self.completion_mode in {"pickup", "dropoff"}, "completion_mode must be 'pickup' or 'dropoff'"
        self.reset_fn = reset_fn
        self._shadow_assigned_by_robot = defaultdict(list)  # rid -> [reservation 1, reservation 2, ...]
        self._shadow_plan_by_robot = defaultdict(list)  # rid -> ["a","a","b","c","b","c", ...]

        # runtime state
        self._step_count: int = 0
        self._last_robot_ids: List[str] = []
        self._last_tasks: List[Task] = []

        # caches
        self._edge_len_cache: Dict[str, float] = {}
        self._route_len_cache: Dict[Tuple[str, str], float] = {}

        # per-tick / across-tick tracking for rewards
        self._prev_assigned_by_robot: Dict[str, Set[str]] = {}
        self._prev_customers_by_robot: Dict[str, Set[str]] = {}
        self._res_created_time: Dict[str, float] = {}  # reservation id → creation time
        self._ever_picked_up: Set[str] = set()
        self._ever_dropped_off: Set[str] = set()

        self.max_wait_delay_s = float(max_wait_delay_s)
        self.max_travel_delay_s = float(max_travel_delay_s)
        self.max_robot_capacity = int(max_robot_capacity)

        self._rng = np.random.default_rng(12345) # for tie-breaker
        self.logger = logger

        self._cum_sum_reward = 0.0
        self._cum_pickups = 0
        self._cum_dropoffs = 0
        self._episode_closed = False

        self.min_episode_steps = int(min_episode_steps)
        self.serve_to_empty = bool(serve_to_empty)
        self.require_seen_reservation = bool(require_seen_reservation)
        self.idle_patience_steps = int(idle_patience_steps)
        self.respect_sumo_end = bool(respect_sumo_end)
        self._idle_streak = 0

        # episode-scoped flags
        self._seen_any_reservation: bool = False
        self._seen_any_customer: bool = False

    # ---------------- Public API ----------------

    # in sumo_rl_rs/environment/rl_controller_adapter.py

    def _reservation_index(self) -> dict[str, object]:
        """id -> reservation object (all states)."""
        try:
            reservations = list(self.sumo.person.getTaxiReservations(0))
        except Exception:
            reservations = []
        out = {}
        for r in reservations:
            rid = str(getattr(r, "id", ""))
            if rid:
                out[rid] = r
        return out

    def _person_to_res_index(self, res_index: dict[str, object]) -> dict[str, str]:
        """
        Map personId -> reservationId. Both the raw person id and the
        stripped version (without any leading 'p') are keeped.
        """
        m = {}
        for rid, r in res_index.items():
            try:
                persons = list(getattr(r, "persons", []))
            except Exception:
                persons = []
            for pid in persons:
                m[str(pid)] = rid
                m[pid.lstrip("p")] = rid
        return m

    def _current_reservation_ids_onboard(self, rid: str, p2r: dict[str, str]) -> list[str]:
        """
        Read device.taxi.currentCustomers (person ids) and map them to reservation ids via p2r.
        device.taxi.currentCustomers: space-separated list of persons that are to be picked up or already on board
        """
        try:
            s = self.sumo.vehicle.getParameter(rid, "device.taxi.currentCustomers") or ""
        except Exception:
            s = ""
        ids = []
        if not s:
            return ids
        seen: set[str] = set()
        out: list[str] = []
        for tok in s.split():
            tok = tok.strip()
            if not tok:
                continue
            # robust to 'p42' and '42'
            res_id = p2r.get(tok) or p2r.get(tok.lstrip("p"))
            if res_id and res_id not in seen:
                out.append(res_id)
                seen.add(res_id)
        return out
    
    def _is_picked(self, res_obj: object) -> bool:
        st = int(getattr(res_obj, "state", 0))
        # 8 == PICKED_UP (state bit)
        return bool(st & 8)

    def _edge_for_pickup(self, res_obj: object) -> str:
        return str(getattr(res_obj, "fromEdge", "")) or ""

    def _edge_for_dropoff(self, res_obj: object) -> str:
        return str(getattr(res_obj, "toEdge", "")) or ""


    def _build_sequence_reset_twice(self, base_ids: list[str]) -> list[str]:
        """
        Build a sequence that forces a full reset:
        For every active reservation r in base_ids, include [r, r].
        SUMO will ignore the first r for customers already picked up.
        """
        seq: list[str] = []
        for r in base_ids:
            seq.extend([r, r])
        return seq



    def _greedy_pd_sequence(
        self,
        rid: str,
        res_ids: list[str],
        res_index: dict[str, object],
        already_picked: set[str],
    ) -> list[str]:
        """
        Build a [id,id,...] sequence using greedy POI selection with the
        constraint "dropoff after pickup unless already picked".
        """
        # build POIs = (res_id, edge_id, kind)
        poi = []
        for r_id in res_ids:
            r = res_index.get(r_id)
            if r is None:
                continue
            # always need a dropoff poi
            poi.append((r_id, self._edge_for_dropoff(r), "dropoff"))
            # pickup only if not already onboard
            if r_id not in already_picked:
                poi.append((r_id, self._edge_for_pickup(r), "pickup"))

        # which POIs may be scheduled next, given what's already scheduled
        def getCandPoi(poi_all, scheduled):
            cand = []
            for p in poi_all:
                if p in scheduled:
                    continue
                r_id, edge, kind = p
                if kind == "pickup":
                    cand.append(p)
                else:  # dropoff
                    # allowed if its pickup is already scheduled OR passenger is already onboard
                    if r_id in already_picked:
                        cand.append(p)
                    else:
                        found_pickup = any((q[0] == r_id and q[2] == "pickup") for q in scheduled)
                        if found_pickup:
                            cand.append(p)
            return cand

        # distance function on edges (road distance)
        def dist(from_edge: str, to_edge: str) -> float:
            if not from_edge or not to_edge:
                return 1e12
            try:
                return float(self.sumo.simulation.getDistanceRoad(from_edge, 0, to_edge, 0, isDriving=True))
            except Exception:
                return 1e12

        # start from taxi's current edge
        try:
            route = self.sumo.vehicle.getRoute(rid)
            current_edge = route[0] if route else ""
        except Exception:
            current_edge = ""

        scheduled: list[tuple[str, str, str]] = []
        total = len(poi)
        while len(scheduled) < total:
            candidates = getCandPoi(poi, scheduled)
            if not candidates:
                # fallback: if constraints block us (shouldn't happen), append any remaining pickups first
                remaining = [p for p in poi if p not in scheduled]
                pickups_first = [p for p in remaining if p[2] == "pickup"] + [p for p in remaining if p[2] == "dropoff"]
                candidates = pickups_first

            # pick the candidate with minimal distance from current edge
            best = min(candidates, key=lambda p: dist(current_edge, p[1])) if current_edge else candidates[0]
            scheduled.append(best)
            current_edge = best[1]

        # return the reservation id sequence (ids may repeat)
        return [p[0] for p in scheduled]


    def taxis_available(self, min_robots: int = 1) -> bool:
        """Return True if at least `min_robots` taxis exist in the sim."""
        try:
            fleet = list(self.sumo.vehicle.getTaxiFleet(-1))
        except Exception:
            fleet = list(self.sumo.vehicle.getIDList())
        return len(fleet) >= max(1, min_robots)

    def wait_for_fleet(self, min_robots: int = 1, max_wait_steps: int = 300) -> int:
        """
        Advance the simulation (no-ops) until there are >= `min_robots` taxis,
        or `max_wait_steps` is reached. Returns number of steps advanced.
        """
        steps = 0
        while steps < max_wait_steps and not self.taxis_available(min_robots):
            # advance sim by one tick without dispatching anything
            self._step()  # uses simulation.step() internally
            steps += 1
        # refresh cached robots/tasks so the next call is consistent
        self._last_robot_ids = self.get_robots()
        self._last_tasks = self.get_tasks()
        return steps


    def reset(self) -> None:
        """Clear internal counters/caches; optionally call external SUMO reset."""
        self._step_count = 0
        self._last_robot_ids.clear()
        self._last_tasks.clear()
        self._edge_len_cache.clear()
        self._route_len_cache.clear()
        self._prev_assigned_by_robot.clear()
        self._prev_customers_by_robot.clear()
        self._res_created_time.clear()
        self._ever_picked_up.clear()
        self._ever_dropped_off.clear()
        self._cum_sum_reward = 0.0
        self._cum_pickups = 0
        self._cum_dropoffs = 0
        self._episode_closed = False
        self._seen_any_reservation = False
        self._seen_any_customer = False
        self._idle_streak = 0


        if self.reset_fn:
            self.reset_fn()

        if self.logger:
            try:
                # bump an internal counter so every reset gets a new ep folder
                self._ep_idx = getattr(self, "_ep_idx", -1) + 1
                # (see RidepoolLogger change below: overwrite=True nukes any old files)
                self.logger.start_episode(self._ep_idx)
            except Exception as e:
                print(f"[logger] start_episode failed: {e}")

        self.wait_for_fleet(min_robots=1, max_wait_steps=300)

    def get_robots(self) -> List[str]:
        """Return ordered list of taxi IDs (robots)."""
        try:
            robots = list(self.sumo.vehicle.getTaxiFleet(-1))
        except Exception:
            robots = list(self.sumo.vehicle.getIDList())
        self._last_robot_ids = robots
        # ensure prev maps exist
        for rid in robots:
            self._prev_assigned_by_robot.setdefault(rid, set())
            self._prev_customers_by_robot.setdefault(rid, set())
        return robots

    def get_tasks(self) -> List[Task]:
        """Return actionable tasks (exclude picked-up), with derived attributes populated."""
        try:
            reservations = list(self.sumo.person.getTaxiReservations(0))
        except Exception:
            reservations = []
        now = self._now()

        if reservations:
            self._seen_any_reservation = True

        # For quick "assigned?" lookup
        assigned_ids: Set[str] = set()
        for r in reservations:
            st = int(getattr(r, "state", self.STATE_REQUESTED))
            if st & self.STATE_ASSIGNED:
                assigned_ids.add(str(getattr(r, "id", "")))

        tasks: List[Task] = []
        for r in reservations:
            rid = str(getattr(r, "id", ""))
            from_edge = str(getattr(r, "fromEdge", ""))
            to_edge = str(getattr(r, "toEdge", ""))
            st = int(getattr(r, "state", self.STATE_REQUESTED))
            t0 = float(getattr(r, "reservationTime", 0.0))
            if rid and rid not in self._res_created_time:
                self._res_created_time[rid] = t0

            # estimate travel time pickup→dropoff
            est_tt = self._estimate_travel_time(from_edge, to_edge)

            # deadlines
            pickup_deadline = t0 + self.max_wait_delay_s
            dropoff_deadline = t0 + est_tt + self.max_travel_delay_s

            # flags
            waiting_time = max(0.0, now - t0)
            is_obsolete = waiting_time > self.max_wait_delay_s and st < self.STATE_PICKED_UP
            is_assigned = (rid in assigned_ids)

            # the task remains in the candidate pool until it is physically picked up
            if st == self.STATE_PICKED_UP:
                continue  # not a "waiting" task for candidates

            tasks.append(Task(
                id=rid,
                fromEdge=from_edge,
                toEdge=to_edge,
                state=st,
                reservationTime=t0,
                estTravelTime=est_tt,
                pickupDeadline=pickup_deadline,
                dropoffDeadline=dropoff_deadline,
                is_obsolete=is_obsolete,
                is_assigned=is_assigned,
            ))

        self._last_tasks = tasks
        return tasks

    def _estimate_travel_time(self, from_edge: str, to_edge: str) -> float:
        """ ETA (sec) from pickup to dropoff using SUMO routing."""
        if not from_edge or not to_edge:
            return 0.0
        try:
            r = self.sumo.simulation.findRoute(from_edge, to_edge)
            if hasattr(r, "travelTime"):
                return float(getattr(r, "travelTime"))
        except Exception:
            pass
        # Fallback estimate: travel_time ≈ distance / 10 m/s (≈36 km/h) if SUMO routing is unavailable.

        dist = self._road_distance(from_edge, to_edge)
        return float(dist / 10.0) if np.isfinite(dist) else 0.0


    def _person_prefixed(self, pid: str) -> str:
        s = str(pid)
        return s if s.startswith("p") else f"p{s}"

    def _res_to_person_list(self, res_obj: object) -> list[str]:
        """Return persons for a reservation with 'p' prefix."""
        persons = list(getattr(res_obj, "persons", [])) if res_obj is not None else []
        return [self._person_prefixed(p) for p in persons]

    def _seq_to_person_pd(self, seq: list[str], res_index: dict[str, object]) -> str:
        """
        Turn a reservation-id sequence (with repeated IDs) into person-level PU/DO tokens.
        First occurrence of a reservation id := PU; second := DO.
        For multi-person reservations, list all persons for that reservation at each occurrence.
        Example token: 'p12:PU+p13:PU' | 'p12:DO+p13:DO' | ...
        """
        seen: dict[str, int] = {}
        tokens: list[str] = []
        for r in seq:
            k = seen.get(r, 0)
            seen[r] = k + 1
            action = "PU" if k == 0 else "DO"
            ps = self._res_to_person_list(res_index.get(r))
            # If the reservation has no persons (edge-case), still log something meaningful
            if not ps:
                ps = [f"r{r}"]
            tokens.append("+".join(f"{p}:{action}" for p in ps))
        return "|".join(tokens)


    def get_candidate_lists(self, K: Optional[int] = None) -> List[List[int]]:
        """
        For each robot, return up to K nearest tasks by pickup distance,
        filtered by road-network distance <= vicinity_m.
        Output indices reference the *most recent* get_tasks() list.
        """
        if not self._last_robot_ids:
            self.get_robots()
        if not self._last_tasks:
            self.get_tasks()

        K = self.k_max if K is None else int(K)
        tasks = self._last_tasks
        tasks = [t for t in tasks if not t.is_obsolete and not t.is_assigned]  # <- keep only viable tasks
        self._last_tasks = tasks  # keep consistent indices

        # robot "position" proxy: first edge of current route
        robot_first_edge: Dict[str, str] = {}
        for rid in self._last_robot_ids:
            try:
                route = self.sumo.vehicle.getRoute(rid)
                if route:
                    robot_first_edge[rid] = route[0]
            except Exception:
                pass

        cand_lists: List[List[int]] = []
        for rid in self._last_robot_ids:
            r_edge = robot_first_edge.get(rid)
            if r_edge is None or not tasks:
                cand_lists.append([])
                continue

            dist_idx: List[Tuple[float, int]] = []
            for j, t in enumerate(tasks):
                if not t.fromEdge:
                    continue
                d = self._road_distance(r_edge, t.fromEdge)
                if np.isfinite(d) and d <= self.vicinity_m:
                    dist_idx.append((float(d), j))
            dist_idx.sort(key=lambda x: x[0])
            cand_lists.append([j for _, j in dist_idx[:K]])
        
   
        if self.logger:
            tnow = self._now()
            res_index = self._reservation_index()  # to resolve persons per reservation
            ids = [t.id for t in self._last_tasks]
            for rid, cidx in zip(self._last_robot_ids, cand_lists):
                # slots are 0..K_i-1
                slots = list(range(len(cidx)))
                # reservation IDs for these candidates
                res_ids = [ids[j] for j in cidx]
                # persons (joined by '+') and PD sequence per candidate ('pX:PU+...+pY:PU+pX:DO+...+pY:DO')
                persons_joined: list[str] = []
                pd_seq_joined: list[str] = []
                for r in res_ids:
                    ps = self._res_to_person_list(res_index.get(r))
                    persons_joined.append("+".join(ps) if ps else "")
                    # PD tokens for this candidate only
                    if ps:
                        pd_tokens = [f"{p}:PU" for p in ps] + [f"{p}:DO" for p in ps]
                        pd_seq_joined.append("+".join(pd_tokens))
                    else:
                        pd_seq_joined.append("")
                # write
                self.logger.log_candidates(
                    tnow,
                    rid,
                    slots,
                    res_ids,
                    persons_joined,   # NEW
                    pd_seq_joined,    # NEW
                )
        self._last_cand_lists = cand_lists
        self._last_task_ids = [t.id for t in self._last_tasks]

        return cand_lists

    def _resolve_assignment_conflicts(
        self,
        robots: List[str],
        chosen: List[Optional[str]],
        tasks_list: List[Task],
    ) -> tuple[List[Optional[str]], Dict[str, str]]:
        """
        For each reservation chosen by multiple taxis, keep exactly one winner:
        - highest remaining capacity wins,
        - ties broken at random (stable via self._rng).
        Returns:
        (resolved_assignments, winners_map) where winners_map is {res_id: rid_winner}.
        """
        # Group: res_id -> [rid, ...]
        buckets: Dict[str, List[str]] = {}
        for rid, res in zip(robots, chosen):
            if res is None:
                continue
            buckets.setdefault(str(res), []).append(rid)

        # Early exit: nothing to resolve
        if not buckets:
            return chosen, {}

        # Remaining capacity per taxi
        rem_cap: Dict[str, int] = {rid: self._remaining_capacity(rid) for rid in robots}

        winners: Dict[str, str] = {}
        for res_id, rids in buckets.items():
            if len(rids) == 1:
                winners[res_id] = rids[0]
                continue

            # Pick taxis with max remaining capacity
            caps = [(rid, rem_cap.get(rid, 0)) for rid in rids]
            max_cap = max(c for _, c in caps)
            best = [rid for rid, c in caps if c == max_cap]

            # Tie-break at random using controller RNG
            if len(best) == 1:
                winners[res_id] = best[0]
            else:
                idx = int(self._rng.integers(0, len(best)))
                winners[res_id] = best[idx]

            # Conflict log (the actual winner is used)
            if self.logger:
                self.logger.log_conflict(
                    self._now(), res_id, rids, [rem_cap[r] for r in rids], winners[res_id]
                )

        # Build resolved list: losers -> None
        resolved: List[Optional[str]] = []
        for rid, res in zip(robots, chosen):
            if res is None:
                resolved.append(None)
            else:
                res_id = str(res)
                resolved.append(res if winners.get(res_id) == rid else None)

        return resolved, winners

    def apply_and_step(self, assignments: Sequence[Optional[Union[int, str]]]) -> Dict[str, Any]:
        """
        Apply assignments (aligned with self.get_robots() order), then advance SUMO one step.
        """

        
        robots = self._last_robot_ids or self.get_robots()
        res_index = self._reservation_index()
        p2r = self._person_to_res_index(res_index)

        # 1) resolve incoming choices -> reservation ids (or None)
        tasks = self._last_tasks or self.get_tasks()
        idx_to_res_id = [t.id for t in tasks]
        cand_lists = getattr(self, "_last_cand_lists", None)

        if self.logger:
            self.logger.log_debug(
                self._now(), "apply-input",
                {"robots": list(robots),
                "assignments_raw": list(assignments[:len(robots)]),
                "cand_counts": [len(cl) for cl in getattr(self, "_last_cand_lists", [])]}
            )
            self.logger.log_debug(self._now(), "apply-len-types", {
                "len_assignments": len(assignments),
                "types": [type(a).__name__ for a in assignments],
            })


        chosen: list[Optional[str]] = []

        for ridx, a in enumerate(assignments[:len(robots)]):
            if a is None:
                chosen.append(None)
                continue
            if isinstance(a, int) and cand_lists is not None:
                clist = cand_lists[ridx] if ridx < len(cand_lists) else []
                if a < len(clist):
                    task_idx = clist[a]
                    chosen.append(idx_to_res_id[task_idx])
                    continue
                chosen.append(None)
                continue
            else:
                chosen.append(str(a))


        if self.logger:
            self.logger.log_debug(self._now(), "apply-mapped", {
                "chosen_res_ids": chosen,  # None or reservation ids per robot
            })

        # 1b) resolve conflicts so only one taxi wins each reservation
        chosen, winners = self._resolve_assignment_conflicts(list(robots), chosen, tasks)

        if self.logger:
            self.logger.log_debug(self._now(), "apply-winners", {
                "winners": winners,  # {res_id: rid}
            })

        # --- Build per-step "owners" map to guarantee global uniqueness ---
        # --- Build exclusive owners map: onboard > winner(this tick) > sticky(previous plan) ---
        onboard_by_res: Dict[str, str] = {}
        for rid0 in robots:
            for res_id in self._current_reservation_ids_onboard(rid0, p2r):
                onboard_by_res[res_id] = rid0

        sticky_by_res: Dict[str, str] = {}
        for rid0 in robots:
            prev_ids_unique = list(dict.fromkeys(self._shadow_plan_by_robot.get(rid0, [])))
            for r in prev_ids_unique:
                if r not in onboard_by_res and r not in sticky_by_res:
                    sticky_by_res[r] = rid0

        owners: Dict[str, str] = dict(sticky_by_res)
        for r, rid_win in winners.items():
            if r not in onboard_by_res:        # never override onboard
                owners[r] = rid_win
        owners.update(onboard_by_res)           # onboard has highest precedence


        # 2) build/dispatch per taxi sequence
        for rid, res_id in zip(robots, chosen):
            prev_seq = list(self._shadow_plan_by_robot.get(rid, []))
            prev_ids = list(dict.fromkeys(prev_seq))
            onboard_ids = self._current_reservation_ids_onboard(rid, p2r)

            # base set: prev uniques ∪ onboard ∪ {new}
            base_ids = list(dict.fromkeys(prev_ids + onboard_ids))
            if res_id and res_id not in base_ids:
                base_ids.append(res_id)

            # keep only reservations whose effective owner is this taxi
            base_ids = [r for r in base_ids if owners.get(r) == rid]

            # FULL RESET strategy: every active reservation appears twice
            seq = self._build_sequence_reset_twice(base_ids)

            # log pre-dispatch
            try:
                raw_cc = self.sumo.vehicle.getParameter(rid, "device.taxi.currentCustomers") or ""
            except Exception:
                raw_cc = ""
            if self.logger:
                seq_pd = self._seq_to_person_pd(seq, res_index)  # <-- NEW
                self.logger.log_dispatch(
                    self._now(), rid, prev_seq, base_ids, seq, raw_cc,  # existing args
                    seq_pd=seq_pd,  # <-- NEW named arg
                    notes="pre-dispatch"
                )
            # avoid spamming SUMO if unchanged
            if seq == prev_seq:
                if self.logger:
                    self.logger.log_dispatch(
                        self._now(), rid, prev_seq, base_ids, seq, raw_cc,
                        seq_pd=self._seq_to_person_pd(seq, res_index),
                        notes="skip-unchanged"
                    )
                continue

            try:
                self.sumo.vehicle.dispatchTaxi(rid, seq)
                self._shadow_plan_by_robot[rid] = seq
            except Exception as e:
                if self.logger:
                    seq_pd = self._seq_to_person_pd(seq, res_index)
                    self.logger.log_dispatch(
                        self._now(), rid, prev_seq, base_ids, seq, raw_cc,
                        seq_pd=seq_pd,
                        notes=f"dispatch-error:{e}"
                    )
                # keep previous plan on failure

        # 3) advance simulation and clean the shadow
        self._step()
        self._refresh_shadow_plan()

        # 4) rewards + logs
        per_robot, terms = self._compute_rewards_per_robot()
        if self.logger:
            tnow = self._now()
            total = 0.0
            pickups = dropoffs = 0
            for rid in (self._last_robot_ids or self.get_robots()):
                r = float(per_robot.get(rid, 0.0))
                total += r
                self.logger.log_rewards(tnow, rid, r, terms.get(rid, {}))
                comp = int(max(0, terms.get(rid, {}).get("completion", 0.0)))
                if self.completion_mode == "pickup":
                    pickups += comp
                else:
                    dropoffs += comp
            self.logger.log_ts_reward(total, pickups, dropoffs)
            idle, enr, occ, pocc = self._fleet_state_counts()
            self.logger.log_fleet_counts(tnow, idle, enr, occ, pocc)

            # accumulate totals for episode_totals.csv
            self._cum_sum_reward += total
            self._cum_pickups  += pickups
            self._cum_dropoffs += dropoffs

            # if episode is done, write a row to episode_totals.csv once
            done = self.is_episode_done()
            if done and self.logger and not self._episode_closed:
                self.logger.end_episode(
                    sum_reward=self._cum_sum_reward,
                    n_pickups=self._cum_pickups,
                    n_dropoffs=self._cum_dropoffs,
                    duration=float(self._step_count),
                )
                self._episode_closed = True


        total = float(sum(per_robot.values()))
        return {"per_robot": per_robot, "sum_reward": total, "terms": terms}

    def _refresh_shadow_plan(self) -> None:
        """Prune sequence entries for finished/onboard removed reservations."""
        # alive reservations
        try:
            alive = {str(getattr(r, "id", "")) for r in self.sumo.person.getTaxiReservations(0)}
        except Exception:
            alive = set()

        # keep ids that still exist (SUMO often removes finished ones)
        for rid, seq in list(self._shadow_plan_by_robot.items()):
            if not seq:
                continue
            kept = [r for r in seq if (not alive or r in alive)]
            self._shadow_plan_by_robot[rid] = kept

    # Helper: is there any work left right now?
    def _no_work_left(self) -> bool:
        # any live reservations?
        try:
            if list(self.sumo.person.getTaxiReservations(0)):
                return False
        except Exception:
            pass
        # any onboard or pending plans?
        for rid in (self._last_robot_ids or self.get_robots()):
            if self._get_current_customers(rid):
                return False
            if self._shadow_plan_by_robot.get(rid, []):
                return False
        # does SUMO expect more entities (vehicles/persons) to still depart?
        try:
            if int(self.sumo.simulation.getMinExpectedNumber()) > 0:
                return False
        except Exception:
            pass
        return True

    def _no_active_work(self) -> bool:
        try:
            if self.sumo.person.getTaxiReservations(0):
                return False
        except Exception:
            pass
        # any onboard or pending plan?
        for rid in (self._last_robot_ids or self.get_robots()):
            if self._get_current_customers(rid):
                return False
            if self._shadow_plan_by_robot.get(rid):
                return False
        return True

    # --- replace is_episode_done() ---
    def is_episode_done(self) -> bool:
        # 1) hard cap by steps (if provided)
        if self.max_steps is not None and self._step_count >= self.max_steps:
            return True

        now = self._now()

        # 2) align to SUMO <time end="..."> if requested
        if self.respect_sumo_end:
            try:
                end_t = float(self.sumo.simulation.getEndTime())
                if end_t > 0 and now >= end_t:
                    return True
            except Exception:
                pass

        # 3) classic TraCI empty check (usually useless with persistent taxis)
        try:
            if int(self.sumo.simulation.getMinExpectedNumber()) == 0:
                return True
        except Exception:
            pass

        # 4) no reservations + fleet idle for a while -> done
        try:
            alive_res = list(self.sumo.person.getTaxiReservations(0))
            n_res = len(alive_res)
        except Exception:
            n_res = 0

        idle, en_route, occupied, pickup_occupied = self._fleet_state_counts()

        if n_res == 0 and en_route == 0 and occupied == 0 and pickup_occupied == 0:
            self._idle_streak += 1
        else:
            self._idle_streak = 0

        if self._step_count >= self.min_episode_steps and self._idle_streak >= self.idle_patience_steps:
            return True

        return False



    # ---------------- Internals ----------------

    def _step(self) -> None:
        """Advance the SUMO simulation one step."""
        try:
            self.sumo.simulation.step()
        except Exception:
            try:
                self.sumo.simulationStep()
            except Exception:
                pass
        self._step_count += 1

    # ---- reward helpers ----

    def _compute_rewards_per_robot(self) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Build per-robot rewards.
        Returns (rewards, terms_per_robot) where terms contain each component for debugging.
        """
        robots = self._last_robot_ids or self.get_robots()
        now = self._now()

        # Snapshot assignment & customers for this tick
        cur_assigned_by_robot: Dict[str, Set[str]] = {rid: set(self._get_assigned_reservations(rid)) for rid in robots}
        cur_customers_by_robot: Dict[str, Set[str]] = {rid: set(self._get_current_customers(rid)) for rid in robots}

        # Track "picked up" events (customer newly onboard this tick)
        picked_up_ids_by_robot: Dict[str, Set[str]] = {}
        dropped_ids_by_robot: Dict[str, Set[str]] = {}

        for rid in robots:
            prev_cust = self._prev_customers_by_robot.get(rid, set())
            cur_cust = cur_customers_by_robot[rid]
            new_pickups = cur_cust - prev_cust
            picked_up_ids_by_robot[rid] = new_pickups
            for pid in new_pickups:
                self._ever_picked_up.add(pid)
            # drop-offs: customers that left since last tick
            dropped = prev_cust - cur_cust
            dropped_ids_by_robot[rid] = dropped
            for did in dropped:
                self._ever_dropped_off.add(did)

        # Abandonment: assigned last tick but removed now without ever being picked up
        abandoned_count_by_robot: Dict[str, int] = {}
        for rid in robots:
            prev_ass = self._prev_assigned_by_robot.get(rid, set())
            cur_ass = cur_assigned_by_robot[rid]
            # Removed from assignment:
            removed = prev_ass - cur_ass
            # Exclude those that were actually picked up this tick (moved from assigned → customers)
            removed_not_picked = {x for x in removed if x not in picked_up_ids_by_robot[rid]}
            # Count as abandoned only if they were never onboard and they didn't complete earlier
            abandoned = {x for x in removed_not_picked if (x not in self._ever_picked_up and x not in self._ever_dropped_off)}
            abandoned_count_by_robot[rid] = len(abandoned)

        # Waiting-time reward at pickup: negative of wait seconds for new pickups
        wait_reward_by_robot: Dict[str, float] = {}
        for rid in robots:
            rew = 0.0
            for pid in picked_up_ids_by_robot[rid]:
                t0 = self._res_created_time.get(pid, now)
                rew += -(now - t0)
            wait_reward_by_robot[rid] = rew

        # Capacity reward: (free capacity) = capacity - current_customers
        capacity_reward_by_robot: Dict[str, float] = {}
        for rid in robots:
            cap = self._get_vehicle_capacity(rid)
            onboard = len(cur_customers_by_robot[rid])
            capacity_reward_by_robot[rid] = float(cap - onboard)

        # Step penalty: constant -1 per robot
        step_penalty = -1.0

        # Completion reward per robot
        completion_by_robot: Dict[str, float] = {}
        if self.completion_mode == "pickup":
            for rid in robots:
                completion_by_robot[rid] = float(len(picked_up_ids_by_robot[rid]))
        else:  # dropoff
            for rid in robots:
                completion_by_robot[rid] = float(len(dropped_ids_by_robot[rid]))

        # Compose rewards
        per_robot: Dict[str, float] = {}
        terms: Dict[str, Dict[str, float]] = {}
        for rid in robots:
            cap = capacity_reward_by_robot[rid]
            abandoned = float(abandoned_count_by_robot[rid])
            waitr = wait_reward_by_robot[rid]
            comp = completion_by_robot[rid]
            r = cap + step_penalty + (-abandoned) + waitr + comp
            per_robot[rid] = float(r)
            terms[rid] = {
                "capacity": cap,
                "step": step_penalty,
                "abandoned": -abandoned,        # penalty actually added
                "wait_at_pickups": waitr,       # negative seconds
                "completion": comp,             # pickups or dropoffs per tick
            }

        # Update previous snapshots
        self._prev_assigned_by_robot = cur_assigned_by_robot
        self._prev_customers_by_robot = cur_customers_by_robot

        return per_robot, terms

    # ---- SUMO helpers ----
    # assigned reservations: use the reservations API (state=4) + optional shadow list
    def _get_assigned_reservations(self, rid: str) -> List[str]:
        seq = self._shadow_plan_by_robot.get(rid, [])
        # unique in order (ids may appear twice for PU/DO)
        return list(dict.fromkeys(seq))


    def _get_current_customers(self, rid: str) -> List[str]:
        """Read current customers onboard a taxi (best-effort)."""
        # Preferred: string space-separated reservation ids
        try:
            s = self.sumo.vehicle.getParameter(rid, "device.taxi.currentCustomers")
            if s:
                return [x for x in s.split() if x]
        except Exception:
            pass
        # Fallback: if TraCI exposes person IDs onboard this vehicle
        try:
            # Some builds have vehicle.getPersonIDList(vehId)
            ids = list(self.sumo.vehicle.getPersonIDList(rid))  # type: ignore[attr-defined]

            if ids:
                self._seen_any_customer = True

            return [str(x) for x in ids]
        except Exception:
            return []

    def _get_vehicle_capacity(self, rid: str) -> int:
        """Read person capacity; fall back to default_capacity on failure."""
        try:
            # SUMO has getPersonCapacity for vehicles in recent versions
            cap = int(self.sumo.vehicle.getPersonCapacity(rid))  # type: ignore[attr-defined]
            if cap > 0:
                return cap
        except Exception:
            pass
        return self.default_capacity

    def _remaining_capacity(self, rid: str) -> int:
        """Free seats right now for taxi rid."""
        cap = self._get_vehicle_capacity(rid)
        onboard = len(self._get_current_customers(rid))
        return max(0, cap - onboard)


    def _now(self) -> float:
        try:
            return float(self.sumo.simulation.getTime())
        except Exception:
            return float(self._step_count)

    # ---- distance helpers for vicinity ----

    def _edge_length(self, edge_id: str) -> float:
        if not edge_id:
            return float("inf")
        if edge_id in self._edge_len_cache:
            return self._edge_len_cache[edge_id]
        try:
            L = float(self.sumo.edge.getLength(edge_id))
        except Exception:
            L = 0.0
        self._edge_len_cache[edge_id] = L
        return L

    def _road_distance(self, from_edge: str, to_edge: str) -> float:
        """
        Best-effort road-network distance via simulation.findRoute().
        Falls back to sum of edge lengths if needed; cached for speed.
        """
        if not from_edge or not to_edge:
            return float("inf")
        key = (from_edge, to_edge)
        if key in self._route_len_cache:
            return self._route_len_cache[key]

        dist = float("inf")
        try:
            route = self.sumo.simulation.findRoute(from_edge, to_edge)
            edges = getattr(route, "edges", None)
            if edges:
                dist = float(sum(self._edge_length(e) for e in edges))
            else:
                if hasattr(route, "length"):
                    dist = float(getattr(route, "length"))
                elif hasattr(route, "travelTime"):
                    dist = float(getattr(route, "travelTime"))  # last-resort proxy
        except Exception:
            if from_edge == to_edge:
                dist = 0.0
            else:
                dist = self._edge_length(from_edge) / 2.0 + self._edge_length(to_edge) / 2.0

        self._route_len_cache[key] = dist
        return dist

    def _fleet_state_counts(self) -> tuple[int,int,int,int]:
        """
        Returns (idle, en_route, occupied, pickup_occupied)
        Heuristic:
        - idle: no onboard persons and no planned reservations
        - en_route: no onboard, but has planned reservations (heading to pickup)
        - occupied: has onboard AND no remaining unpicked in plan (driving to dropoffs)
        - pickup_occupied: has onboard AND still has at least one unpicked in plan
        """
        robots = self._last_robot_ids or self.get_robots()
        idle = en_route = occupied = pickup_occupied = 0

        # quick lookup: reservations that are picked
        res_index = self._reservation_index()
        def _is_picked_res(rid: str) -> bool:
            r = res_index.get(rid)
            st = int(getattr(r, "state", 0)) if r is not None else 0
            return bool(st & self.STATE_PICKED_UP)

        for rid in robots:
            onboard = len(self._get_current_customers(rid))  
            plan = self._shadow_plan_by_robot.get(rid, [])
            if onboard == 0:
                if plan:
                    en_route += 1
                else:
                    idle += 1
            else:
                # any unpicked reservations still in plan?
                has_unpicked = any(not _is_picked_res(r) for r in dict.fromkeys(plan))
                if has_unpicked:
                    pickup_occupied += 1
                else:
                    occupied += 1
        return idle, en_route, occupied, pickup_occupied
