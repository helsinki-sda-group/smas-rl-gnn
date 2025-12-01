import torch as th
import torch.nn as nn
from typing import Any, Dict, Optional, List, Literal, cast
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .actor_critic import EgoActorCritic  # pluggable GNN-based actor–critic


class DictPassthroughExtractor(BaseFeaturesExtractor):
    """
    Captures the raw dictionary observation for the policy and returns a
    dummy feature tensor to satisfy the interface expected by SB3's
    ActorCriticPolicy.
    """

    def __init__(self, observation_space: spaces.Dict):
        # features_dim must be > 0
        super().__init__(observation_space, features_dim=1)
        self.last_obs: Optional[Dict[str, th.Tensor]] = None

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        self.last_obs = obs   # observation tensors already include a batch dimension
        any_tensor = next(iter(obs.values()))
        B = any_tensor.shape[0]
        return th.ones((B, 1), device=any_tensor.device, dtype=any_tensor.dtype)


class RTGNNPolicy(ActorCriticPolicy):
    """
    SB3 policy wrapper around EgoActorCritic.

    IMPORTANT:
      - Env action_space is MultiDiscrete with nvec = [K_max+1] * R
        (last slot is explicit NO-OP per robot).
      - EgoActorCritic keeps producing logits of shape [R, K_max] (no change).
      - Single learnable NO-OP logit (one scalar parameter) is added as the (K_max)-th column.
      - A corresponding mask entry is appended to ensure the NO-OP option is always valid.
    """
    def __init__(
        self,
        *args,
        in_dim: int,
        hidden: int,
        k_max: int,
        backbone: str = "sage",
        critic_aggregation: str = "joint_mean",
        **kwargs,
    ):
        # Use the passthrough extractor to preserve the raw dict observation
        super().__init__(*args, features_extractor_class=DictPassthroughExtractor, **kwargs)
        self.k_max = k_max               # logits from GNN
        self.k_out = k_max + 1           # +1 explicit NO-OP per robot (matches env action space)

        gnn_kwargs: Dict[str, Any] = kwargs.pop("gnn_kwargs", {})

        # normalize aggregation aliases to the model's canonical set
        agg_aliases = {
            "mean": "joint_mean",
            "avg": "joint_mean",
            "attn": "joint_attn",
            "attention": "joint_attn",
            "perrobot": "per_robot",
            "pr": "per_robot",
            "independent": "per_robot",
        }

        _bb_allowed = ("dummy", "sage")
        _agg_allowed = ("per_robot", "joint_mean", "joint_attn")
        if backbone not in _bb_allowed:
            raise ValueError(f"Invalid backbone='{backbone}'. Allowed: {_bb_allowed}")
        if critic_aggregation not in _agg_allowed:
            raise ValueError(f"Invalid critic_aggregation='{critic_aggregation}'. Allowed: {_agg_allowed}")

        bb_lit = cast(Literal["dummy", "sage"], backbone)
        agg_lit = cast(Literal["per_robot", "joint_mean", "joint_attn"], critic_aggregation)

        self.gnn_ac = EgoActorCritic(
            in_dim=in_dim,
            hidden=hidden,
            k_max=k_max,  # keep the original K for the GNN head
            backbone=bb_lit,
            critic_aggregation=agg_lit,
            **gnn_kwargs,
        )

        # # Single learnable NO-OP logit shared across robots and batch.
        self.noop_logit = nn.Parameter(th.tensor(0.0))

    # --- helpers -------------------------------------------------------------

    def _build_batch_outputs(self, obs_b: Dict[str, th.Tensor]) -> tuple[th.Tensor, th.Tensor]:
        """
        Run EgoActorCritic per batch element (SB3 supplies a batch dimension B).
        Returns:
            logits: [B, R, K_max]   (NO-OP column appended later)
            values: [B, 1]         (scalar per env sample, regardless of critic mode)
        """
        # Batch size B inferred from any tensor
        B = next(iter(obs_b.values())).shape[0]
        logits_list: List[th.Tensor] = []
        values_list: List[th.Tensor] = []

        # Process each env sample independently (keeps EgoActorCritic simple)
        for b in range(B):
            obs_one = {k: v[b] for k, v in obs_b.items()}   # strip batch dim
            logits_b, value_b = self.gnn_ac(obs_one)        # logits: [R,K], value: scalar or [R]

            logits_list.append(logits_b)

            # Normalize value to scalar per env sample
            if value_b.dim() == 0:       # scalar (joint_* critic)
                v_b = value_b
            elif value_b.dim() == 1:     # [R] (per_robot critic) -> choose an aggregation
                v_b = value_b.mean()     # or .sum() depending on reward convention
            else:
                v_b = value_b.squeeze()
            values_list.append(v_b)

        logits = th.stack(logits_list, dim=0)               # [B, R, K_max]
        values = th.stack(values_list, dim=0).unsqueeze(-1) # [B, 1]
        return logits, values

    def _append_noop(self, logits: th.Tensor, mask_k: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Append NO-OP column to logits and mask.
        logits: [B,R,K_max], mask_k: [B,R,K_max] (bool or 0/1)
        returns: logits_full: [B,R,K_max+1], mask_full: [B,R,K_max+1] (bool)
        """
        B, R, _K = logits.shape
        # NO-OP logit is a shared scalar parameter used for all robots and all batch elements
        noop_col = self.noop_logit.expand(B, R, 1)
        logits_full = th.cat([logits, noop_col], dim=-1)  # [B,R,K_max+1]

        # Construct the mask by appending an always-valid entry for the NO-OP slot
        if mask_k.dtype != th.bool:
            mask_k = mask_k.bool()
        ones = th.ones((B, R, 1), dtype=th.bool, device=mask_k.device)
        mask_full = th.cat([mask_k, ones], dim=-1)        # [B,R,K_max+1]
        return logits_full, mask_full

    def _dist_from_logits(self, logits: th.Tensor, mask: th.Tensor):
        """
        Construct the SB3 action distribution.
        Inputs:
            logits: [B, R, K_out]
            mask:   [B, R, K_out] (bool)
        The logits are flattened along the last two dimensions to match MultiDiscrete semantics.
        """

        logits = logits.masked_fill(~mask, -1e9)
        B = logits.size(0)
        logits_flat = logits.reshape(B, -1)                 # [B, R*K_out]
        return self.action_dist.proba_distribution(action_logits=logits_flat)

    # --- main SB3 hooks ------------------------------------------------------

    def forward(self, obs: Any, deterministic: bool = False):
        # Pass the observation through the features extractor to obtain the dict with batch dimension
        _ = self.extract_features(obs, features_extractor=self.features_extractor)

        obs_dict_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
        assert obs_dict_b is not None, "Features extractor did not capture obs dict"

        logits_k, values = self._build_batch_outputs(obs_dict_b)           # [B,R,K_max], [B,1]
        mask_k = obs_dict_b["cand_mask"]                                   # [B,R,K_max]
        logits, mask = self._append_noop(logits_k, mask_k)                 # [B,R,K_max+1] each

        dist = self._dist_from_logits(logits, mask)
        actions = dist.get_actions(deterministic=deterministic)            # [B, R]
        log_prob = dist.log_prob(actions)                                  # [B]

        return actions, values, log_prob

    def evaluate_actions(self, obs: Any, actions: th.Tensor):
        """
        Evaluation method used by SB3 during training to compute log-probabilities,
        entropy, and value estimates. Must be consistent with `forward()` and the
        distribution used during action sampling.
        """

        _ = self.extract_features(obs, features_extractor=self.features_extractor)
        obs_dict_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
        assert obs_dict_b is not None

        logits_k, values = self._build_batch_outputs(obs_dict_b)           # [B,R,K_max], [B,1]
        mask_k = obs_dict_b["cand_mask"]                                   # [B,R,K_max]
        logits, mask = self._append_noop(logits_k, mask_k)                 # [B,R,K_max+1]

        dist = self._dist_from_logits(logits, mask)
        log_prob = dist.log_prob(actions)                                  # [B]
        entropy = dist.entropy()                                           # [B]
        return values, log_prob, entropy

    def predict_values(self, obs: Any) -> th.Tensor:
        _ = self.extract_features(obs, features_extractor=self.features_extractor)
        obs_dict_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
        assert obs_dict_b is not None
        _, values = self._build_batch_outputs(obs_dict_b)                  # [B,1]
        return values
