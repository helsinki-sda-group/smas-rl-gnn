import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from typing import List, Optional, Dict, Any

from rt_gnn_rl.graphs import build_padded_ego_batch  # implemented below
from rt_gnn_rl.policy.action_context import pop_latest_policy_step
from rt_gnn_rl.policy.timing_context import pop_latest_policy_timing

class RidepoolRTEnv(gym.Env):
    """
    Gymnasium environment wrapping the RLControllerAdapter and SUMO Taxi simulation.

    Observation:
        Dict of padded ego-graphs (one per robot) produced by build_padded_ego_batch.
    Action:
        MultiDiscrete([K_max+1] * R) – candidate slot per robot, with the last slot
        reserved for a no-op action.
    Control frequency:
        Macro-decisions every `decision_dt` simulation seconds (no-op in between).
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        controller,
        *,
        R: int,
        K_max: int,
        N_max: int,
        E_max: int,
        F: int,
        G: int,
        feature_fn,
        global_stats_fn=None,
        decision_dt: int = 15, # seconds between policy decisions (1 = every second)
        two_hop: bool = False,
        two_hop_directed: bool = False,
        normalize_features: bool = False,
        use_edge_rt: bool = False,
        edge_feat_dim: int = 0,
        edge_features: Optional[List[str]] = None,
    ):
        super().__init__()
        self.controller = controller
        self.R, self.K_max = int(R), int(K_max)
        self.N_max, self.E_max = int(N_max), int(E_max)
        self.F, self.G = int(F), int(G)
        self.feature_fn = feature_fn
        self.global_stats_fn = global_stats_fn  # can be None (ignored)
        self.two_hop = bool(two_hop)
        self.two_hop_directed = bool(two_hop_directed)
        self.normalize_features = bool(normalize_features)
        self.use_edge_rt = bool(use_edge_rt)
        self.edge_feat_dim = int(edge_feat_dim)
        self.edge_features = list(edge_features or [])
        self._episode_reward = 0
        self._macro_step = 0
        self.timing_collector = None
        self._pending_obs_build_ns = 0

        # explicit no-op slot at the END of each robot's action vector
        self._noop_index = self.K_max
        # Spaces (fixed shapes; runtime masking handles sparsity)
        self.action_space = spaces.MultiDiscrete([self.K_max+1] * self.R)
        obs_space = {
            "x":            spaces.Box(-np.inf, np.inf, (self.R, self.N_max, self.F), dtype=np.float32),
            "node_mask":    spaces.MultiBinary((self.R, self.N_max)),
            "edge_index":   spaces.Box(0, self.N_max - 1, (self.R, 2, self.E_max), dtype=np.int64),
            "edge_mask":    spaces.MultiBinary((self.R, self.E_max)),
            "cand_idx":     spaces.Box(0, self.N_max - 1, (self.R, self.K_max), dtype=np.int64),
            "cand_mask":    spaces.MultiBinary((self.R, self.K_max)),
            # "global_stats": spaces.Box(-np.inf, np.inf, (self.G,), dtype=np.float32),
        }
        if self.edge_feat_dim > 0:
            obs_space["edge_attr"] = spaces.Box(
                -np.inf,
                np.inf,
                (self.R, self.E_max, self.edge_feat_dim),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(obs_space)

        # cached per-robot candidate mapping: slot -> task_id (string)
        self._last_cand_task_ids: List[List[Optional[str]]] = [[] for _ in range(self.R)]
        self._last_robot_ids: List[Optional[str]] = [None] * self.R

        # macro-decision control
        self.decision_dt = int(decision_dt)
        assert self.decision_dt >= 1, "decision_dt must be >=1"

    # --- helpers
    def _sync_from_controller(self) -> Dict[str, Any]:
        robots = self.controller.get_robots()
        tasks_viable, cand_lists = self.controller.get_tasks_and_candidate_lists(self.K_max)

        # Trim robots to R; pad with None to keep fixed shapes
        robots = robots[: self.R]
        if len(robots) < self.R:
            robots += [None] * (self.R - len(robots))

        # Ensure there are cand_lists for each of the first R robots
        if len(cand_lists) < self.R:
            cand_lists += [[] for _ in range(self.R - len(cand_lists))]
        cand_lists = cand_lists[: self.R]

        return {"robots": robots, "tasks": tasks_viable, "cand_lists": cand_lists}


    def _build_obs(self):
        snap = self._sync_from_controller()
        robots = snap["robots"]
        tasks = snap["tasks"]
        cand_lists = snap["cand_lists"]

        obs, cand_task_ids = build_padded_ego_batch(
            robots=robots,
            tasks=tasks,
            candidate_lists=cand_lists,  # lists of indices into `tasks`
            N_max=self.N_max, E_max=self.E_max, K_max=self.K_max,
            F=self.F, G=self.G,
            feature_fn=self.feature_fn,
            two_hop=self.two_hop,
            two_hop_directed=self.two_hop_directed,
            normalize_features=self.normalize_features,
            vicinity_m=float(getattr(self.controller, "vicinity_m", 0.0)),
            use_edge_rt=self.use_edge_rt,
            edge_feat_dim=self.edge_feat_dim,
            edge_features=self.edge_features,
            # global_stats_fn=self.global_stats_fn,  # currently unused
        )
        # Save the exact ids used for slots this step (for action mapping)
        self._last_cand_task_ids = cand_task_ids
        self._last_robot_ids = list(robots)

        x = obs["x"]

        
        # print("tasks:", len(tasks))
        # print("cand_lists:", cand_lists)
        # print("cand_task_ids:", cand_task_ids)
        # print("robot features:", x[0,0])
        # print("task features:", x[0,1:5])
        # print("cand_mask:", obs["cand_mask"][0])

        return obs

    # --- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Adapter's reset doesn't take seed; that's fine.
        self.controller.reset()
        self._episode_reward = 0
        self._macro_step = 0
        t_obs_start = time.perf_counter_ns()
        obs = self._build_obs()
        self._pending_obs_build_ns = time.perf_counter_ns() - t_obs_start
        info = {"action_mask": self.action_mask()}
        return obs, info
    
    def _decode(self, action_vec: np.ndarray) -> List[Optional[str]]:
        """
        Map action vector  -> per-robot assignment list
        Each entry action vector[r] is an integer index (not a task id). 
        Decode converts those indices into actual task IDs 
        For example, action_vec = [0, 3, 0, 3, 0], and _last_cand_task_ids[1] = ['0', None, None]
        - slot in [0..K_max-1] and valid: assign that candidate's task id
        - slot == K_max (self._noop_index) OR invalid: None (no-op)
        """
        out: List[Optional[str]] = []
        for r in range(self.R):
            a = int(action_vec[r])
            if a == self._noop_index:
                out.append(None)
                continue
            # guard if fewer than K_max candidates this tick
            if 0 <= a < len(self._last_cand_task_ids[r]):
                out.append(self._last_cand_task_ids[r][a])
            else:
                out.append(None)    # treat out-of-range as no-op
        return out
    
    def noop_action(self) -> np.ndarray:
        """ All robots no-op action. """
        return np.full(self.R, self._noop_index, dtype=np.int64)
    
    def action_mask(self) -> np.ndarray:
        """
        Shape [R, K_max+1]; 1=allowed, 0=blocked.
        Real candidate slots 0..k-1 are allowed for each robot; K_max is always allowed (no-op).
        """
        mask = np.zeros((self.R, self.K_max+1), dtype=np.int8)
        for r, slots in enumerate(self._last_cand_task_ids):
            # allow only the actual, non-None candidates
            for j in range(min(self.K_max, len(slots))):
                if slots[j] is not None:
                    mask[r,j] = 1
            # explicit no-op always valid
            mask[r, self._noop_index] = 1
        return mask

    def _selected_task_margins(self, action_vec: np.ndarray) -> Dict[str, float]:
        payload = pop_latest_policy_step()
        if payload is None:
            self._cached_raw_logits: Dict[str, float] = {}
            return {}

        logits_k = np.asarray(payload.get("logits_k"), dtype=np.float32)
        mask_k = np.asarray(payload.get("mask_k"), dtype=bool)
        if logits_k.ndim == 3:
            logits_k = logits_k[0]
        if mask_k.ndim == 3:
            mask_k = mask_k[0]
        if logits_k.ndim != 2 or mask_k.ndim != 2:
            self._cached_raw_logits = {}
            return {}

        margins: Dict[str, float] = {}
        raw_logits: Dict[str, float] = {}
        num_rows = min(self.R, logits_k.shape[0], mask_k.shape[0])
        for r in range(num_rows):
            rid = self._last_robot_ids[r] if r < len(self._last_robot_ids) else None
            chosen_slot = int(action_vec[r])
            if rid is None or chosen_slot == self._noop_index:
                continue
            if chosen_slot < 0 or chosen_slot >= logits_k.shape[1] or chosen_slot >= mask_k.shape[1]:
                continue
            if not bool(mask_k[r, chosen_slot]):
                continue

            valid_slots = np.flatnonzero(mask_k[r])
            if valid_slots.size == 0 or chosen_slot not in valid_slots:
                continue

            chosen_logit = float(logits_k[r, chosen_slot])
            raw_logits[str(rid)] = chosen_logit
            other_slots = valid_slots[valid_slots != chosen_slot]
            max_other = float(np.max(logits_k[r, other_slots])) if other_slots.size > 0 else 0.0
            margins[str(rid)] = chosen_logit - max_other

        self._cached_raw_logits = raw_logits
        return margins

    def _selected_task_raw_logits(self, action: np.ndarray) -> Dict[str, float]:
        """Raw selected-task logits per robot, keyed by taxi id (string).
        Populated as a side-effect of _selected_task_margins (which pops the
        shared policy-step payload); always call margins first."""
        return getattr(self, "_cached_raw_logits", {})


    def step(self, action):
        """
        One macro-decision:
            (1) apply chosen assignments for this tick
            (2) advance (decision_dt-1) ticks as no-ops
            (3) sum rewards; break early if episode ends

            Action is a vector of indices, e.g. [0 3 0 3 0] where 3 is no-op index in this example
            Assignments are task ids, e.g. ['0', None, '0', None, '0']
        """
        action = np.asarray(action, dtype=np.int64)
        total_reward = 0.0
        terminated = False
        truncated = False
        last_info: Dict[str, Any] = {}

        sum_terms = {
            "capacity": 0.,
            "step": 0.,
            "deadline": 0.,
            "wait": 0.,
            "travel": 0.,
            "completion": 0.,
            "nonserved": 0.,
        }

        t_decision_start = time.perf_counter_ns()
        obs_build_ns = int(max(0, self._pending_obs_build_ns))
        env_pre_controller_start_ns = time.perf_counter_ns()

        # (1) apply chosen assignments now
        decision_action_mask = self.action_mask()
        n_candidate_pairs = int(np.sum(decision_action_mask[:, : self.K_max]))
        n_nonempty_robots = int(np.sum(np.any(decision_action_mask[:, : self.K_max] > 0, axis=1)))

        t_map_start = time.perf_counter_ns()
        assignments = self._decode(action)
        action_mapping_ns = time.perf_counter_ns() - t_map_start

        task_counts: Dict[str, int] = {}
        n_proposals = 0
        for assigned in assignments:
            if assigned is None:
                continue
            n_proposals += 1
            key = str(assigned)
            task_counts[key] = task_counts.get(key, 0) + 1
        n_bid_tasks = int(len(task_counts))
        n_conflicting_tasks = int(sum(1 for cnt in task_counts.values() if cnt > 1))

        selected_task_margins = self._selected_task_margins(action)
        selected_task_raw_logits = self._selected_task_raw_logits(action)
        env_pre_controller_ns = time.perf_counter_ns() - env_pre_controller_start_ns

        proposal_ns = 0
        resolution_ns = 0
        pre_step_sync_ns = 0
        simulation_ns = 0
        post_step_logging_ns = 0

        t_resolve_start = time.perf_counter_ns()
        step_out = self.controller.apply_and_step(
            assignments,
            allow_redispatch=True,
            selection_margins=selected_task_margins,
            selection_raw_logits=selected_task_raw_logits,
            action_slots=action.tolist(),
            noop_index=self._noop_index,
            final_action_mask=decision_action_mask,
            policy_robot_ids=list(self._last_robot_ids),
        )  # controller aligns with its robot order
        t_apply_end = time.perf_counter_ns()

        controller_timing = getattr(self.controller, "_last_timing", {}) or {}
        proposal_ns_controller = int(controller_timing.get("proposal_ns", 0))
        proposal_ns = int(proposal_ns_controller)
        resolution_ns = int(controller_timing.get("resolution_ns", 0))
        commit_dispatch_ns = int(controller_timing.get("commit_dispatch_ns", 0))
        pre_step_sync_ns = int(controller_timing.get("pre_step_sync_ns", 0))
        simulation_ns = int(controller_timing.get("simulation_ns", 0))
        post_step_logging_ns = int(controller_timing.get("post_step_logging_ns", 0))

        if proposal_ns <= 0 and resolution_ns <= 0 and simulation_ns <= 0:
            # Fallback only if controller-level phase timings are unavailable.
            proposal_ns = max(0, int(t_resolve_start - t_decision_start))
            resolution_ns = 0
            simulation_ns = max(0, int(t_apply_end - t_resolve_start))

         # Expect dict like {"per_robot": {...}, "sum_reward": float, "terms": {...}}
        total_reward += float(step_out.get("sum_reward", 0.0))
        terminated = bool(self.controller.is_episode_done())
        last_info = {k: v for k, v in step_out.items() if k != "sum_reward"}

        terms_out = step_out.get("terms",{})
        for rid in terms_out:
            for key in sum_terms:
                sum_terms[key] += terms_out[rid].get(key, 0.)

        self._macro_step +=1

        # print("action = ", action)
    
        # (2) macro no-op rollout
        steps_done = 1
        while (not terminated) and steps_done < self.decision_dt:
            noop = [None] * self.R
            step_out = self.controller.apply_and_step(noop, allow_redispatch = False)
            loop_timing = getattr(self.controller, "_last_timing", {}) or {}
            simulation_ns += int(loop_timing.get("simulation_ns", 0))
            total_reward += float(step_out.get("sum_reward", 0.0))
            terminated = bool(self.controller.is_episode_done())
            last_info = {k: v for k, v in step_out.items() if k != "sum_reward"}

            terms_out = step_out.get("terms",{})
            for rid in terms_out:
                for key in sum_terms:
                    sum_terms[key] += terms_out[rid].get(key, 0.)

            steps_done += 1

        decision_total_ns = time.perf_counter_ns() - t_decision_start

       
        # (3) build next obs only at the macro boundary
        next_obs_build_ns = 0
        try:
            t_next_obs_start = time.perf_counter_ns()
            obs = self._build_obs()
            next_obs_build_ns = time.perf_counter_ns() - t_next_obs_start
        except Exception:
            obs = {k: np.zeros_like(v) for k,v in self.observation_space.spaces.items()}
            next_obs_build_ns = 0
        self._pending_obs_build_ns = int(max(0, next_obs_build_ns))

        #total_reward = total_reward/100.0 # self.decision_dt

        self._episode_reward += total_reward

        # macro info for reward logging
        # All components are cumulative (not averaged) to match macro_reward
        macro_info = {
            "macro_reward": round(total_reward, 3),
            "macro_capacity": round(sum_terms["capacity"],3),
            "macro_step": round(sum_terms["step"],3),
            "macro_deadline": round(sum_terms["deadline"],3),
            "macro_wait": round(sum_terms["wait"],3),
            "macro_travel": round(sum_terms["travel"],3),
            "macro_completion": round(sum_terms["completion"],3),
            "macro_nonserved": round(sum_terms["nonserved"],3),
        }
        
        info = {
            **last_info,
            "action_mask": self.action_mask(),
            "macro_steps": self._macro_step, # how many sim tick were consumed
        }

        info.update(macro_info)

        if self.controller.logger:
            self.controller.logger.log_macro_step(info)

        if terminated or truncated:
            info["episode_reward"] = self._episode_reward
            info["steps_done"] = steps_done
            self._episode_reward = 0

        info["timing"] = {
            "env_pre_controller_ns": int(env_pre_controller_ns),
            "pre_step_sync_ns": int(pre_step_sync_ns),
            "proposal_ns": int(proposal_ns),
            "resolution_ns": int(resolution_ns),
            "commit_dispatch_ns": int(commit_dispatch_ns),
            "simulation_ns": int(simulation_ns),
            "post_step_logging_ns": int(post_step_logging_ns),
            "decision_total_ns": int(decision_total_ns),
            "other_ns": 0,
            "n_candidate_pairs": int(n_candidate_pairs),
            "n_nonempty_robots": int(n_nonempty_robots),
            "n_proposals": int(n_proposals),
            "n_bid_tasks": int(n_bid_tasks),
            "n_conflicting_tasks": int(n_conflicting_tasks),
            "gnn_action_mapping_ns": int(action_mapping_ns),
            "gnn_obs_build_ns": int(obs_build_ns),
            "controller_proposal_ns": int(proposal_ns_controller),
        }

        policy_timing = pop_latest_policy_timing()
        if policy_timing is not None:
            info["timing"].update(policy_timing)

        # Compose shared outer proposal time with available components.
        if policy_timing is not None:
            info["timing"]["proposal_ns"] = int(
                max(0, obs_build_ns)
                + int(max(0, policy_timing.get("gnn_tensor_prepare_ns", 0)))
                + int(max(0, policy_timing.get("gnn_policy_total_ns", 0)))
                + int(max(0, policy_timing.get("gnn_action_ns", 0)))
                + int(max(0, action_mapping_ns))
                + int(max(0, proposal_ns_controller))
            )
        else:
            info["timing"]["proposal_ns"] = int(
                max(0, obs_build_ns)
                + int(max(0, action_mapping_ns))
                + int(max(0, proposal_ns_controller))
            )

        info["timing"]["other_ns"] = int(
            max(
                0,
                int(info["timing"]["decision_total_ns"])
                - int(info["timing"].get("env_pre_controller_ns", 0))
                - int(info["timing"].get("pre_step_sync_ns", 0))
                - int(info["timing"]["proposal_ns"])
                - int(info["timing"]["resolution_ns"])
                - int(info["timing"].get("commit_dispatch_ns", 0))
                - int(info["timing"]["simulation_ns"]),
                - int(info["timing"].get("post_step_logging_ns", 0))
            )
        )


        return obs, total_reward, terminated, truncated, info
