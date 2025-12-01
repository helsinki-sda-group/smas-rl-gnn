import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Optional, Dict, Any

from rt_gnn_rl.graphs import build_padded_ego_batch  # implemented below

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
    ):
        super().__init__()
        self.controller = controller
        self.R, self.K_max = int(R), int(K_max)
        self.N_max, self.E_max = int(N_max), int(E_max)
        self.F, self.G = int(F), int(G)
        self.feature_fn = feature_fn
        self.global_stats_fn = global_stats_fn  # can be None (ignored)

        # explicit no-op slot at the END of each robot's action vector
        self._noop_index = self.K_max
        # Spaces (fixed shapes; runtime masking handles sparsity)
        self.action_space = spaces.MultiDiscrete([self.K_max+1] * self.R)
        self.observation_space = spaces.Dict({
            "x":            spaces.Box(-np.inf, np.inf, (self.R, self.N_max, self.F), dtype=np.float32),
            "node_mask":    spaces.MultiBinary((self.R, self.N_max)),
            "edge_index":   spaces.Box(0, self.N_max - 1, (self.R, 2, self.E_max), dtype=np.int64),
            "edge_mask":    spaces.MultiBinary((self.R, self.E_max)),
            "cand_idx":     spaces.Box(0, self.N_max - 1, (self.R, self.K_max), dtype=np.int64),
            "cand_mask":    spaces.MultiBinary((self.R, self.K_max)),
            # "global_stats": spaces.Box(-np.inf, np.inf, (self.G,), dtype=np.float32),
        })

        # cached per-robot candidate mapping: slot -> task_id (string)
        self._last_cand_task_ids: List[List[Optional[str]]] = [[] for _ in range(self.R)]

        # macro-decision control
        self.decision_dt = int(decision_dt)
        assert self.decision_dt >= 1, "decision_dt must be >=1"

    # --- helpers
    def _sync_from_controller(self) -> Dict[str, Any]:
        robots = self.controller.get_robots()
        tasks = self.controller.get_tasks()
        cand_lists = self.controller.get_candidate_lists(self.K_max)

        # Trim robots to R; never append Nones
        robots = robots[: self.R]

        # Ensure there are cand_lists for each of the first R robots
        if len(cand_lists) < self.R:
            cand_lists += [[] for _ in range(self.R - len(cand_lists))]
        cand_lists = cand_lists[: self.R]

        return {"robots": robots, "tasks": tasks, "cand_lists": cand_lists}


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
            # global_stats_fn=self.global_stats_fn,  # currently unused
        )
        # Save the exact ids used for slots this step (for action mapping)
        self._last_cand_task_ids = cand_task_ids
        return obs

    # --- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Adapter's reset doesn't take seed; that's fine.
        self.controller.reset()
        obs = self._build_obs()
        info = {"action_mask": self.action_mask()}
        return obs, info
    
    def _decode(self, action_vec: np.ndarray) -> List[Optional[str]]:
        """
        Map action vector -> per-robot assignment list
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


    def step(self, action):
        """
        One macro-decision:
            (1) apply chosen assignments for this tick
            (2) advance (decision_dt-1) ticks as no-ops
            (3) sum rewards; break early if episode ends
        """
        action = np.asarray(action, dtype=np.int64)
        total_reward = 0.0
        terminated = False
        truncated = False
        last_info: Dict[str, Any] = {}

        # (1) apply chosen assignments now
        assignments = self._decode(action)
        step_out = self.controller.apply_and_step(assignments)  # controller aligns with its robot order
         # Expect dict like {"per_robot": {...}, "sum_reward": float, "terms": {...}}
        total_reward += float(step_out.get("sum_reward", 0.0))
        terminated = bool(self.controller.is_episode_done())
        last_info = {k: v for k, v in step_out.items() if k != "sum_reward"}

        # (2) macro no-op rollout
        steps_done = 1
        while (not terminated) and steps_done < self.decision_dt:
            noop = [None] * self.R
            step_out = self.controller.apply_and_step(noop)
            total_reward += float(step_out.get("sum_reward", 0.0))
            terminated = bool(self.controller.is_episode_done())
            last_info = {k: v for k, v in step_out.items() if k != "sum_reward"}
            steps_done += 1
        
        # (3) build next obs only at the macro boundary
        obs = self._build_obs()

        info = {
            **last_info,
            "action_mask": self.action_mask(),
            "macro_steps": steps_done, # how many sim tick were consumed
        }


        return obs, total_reward, terminated, truncated, info
