# rt_gnn_rl/policy/actor_critic.py
import torch
import torch.nn as nn
from typing import List, Literal, Optional

from .gnn_backbone import DummyBackbone, EgoGraphEncoder


class _BatchView:
    """Lightweight container exposing a .batch vector mapping nodes to graph indices."""
    def __init__(self, batch_vec: torch.Tensor):
        self.batch = batch_vec

    
class _DummyEgoEncoder(nn.Module):
    """
    Adapter around DummyBackbone providing the (h, batch) interface expected by the policy code.
    """

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.mlp = DummyBackbone(in_dim, hidden)

    def encode_graphs(self, x_list: List[torch.Tensor], _ei_list: List[torch.Tensor], _extra=None):
        h_list, batch_ids = [], []
        for i, x in enumerate(x_list):
            h_i = self.mlp(x)  # [n_i, H]
            h_list.append(h_i)
            batch_ids.append(torch.full((x.size(0),), i, dtype=torch.long, device=x.device))
        if len(h_list) == 0:
            # No nodes at all — return empty tensors on CPU-safe defaults
            return (torch.empty(0, 0), _BatchView(torch.empty(0, dtype=torch.long)))
        h = torch.cat(h_list, dim=0)                      # [sum_nodes, H]
        batch = _BatchView(torch.cat(batch_ids, dim=0))   # [sum_nodes]
        return h, batch


class _SAGEEgoEncoder(nn.Module):
    """
    Adapter around EgoGraphEncoder that exposes a unified encode_graphs(...) API
    for the actor–critic.
    """
    def __init__(self, in_dim: int, hidden: int, **gnn_kwargs):
        super().__init__()
        self.impl = EgoGraphEncoder(in_dim, hidden, **gnn_kwargs)

    def encode_graphs(self, x_list: List[torch.Tensor], ei_list: List[torch.Tensor], extra=None):
        h, pyg_batch = self.impl(x_list, ei_list, extra)
        # pyg_batch already has .batch; use it directly.
        return h, pyg_batch


class EgoActorCritic(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        k_max: int,
        backbone: Literal["dummy", "sage"] = "dummy",
        critic_aggregation: Literal["per_robot", "joint_mean", "joint_attn"] = "joint_attn",
        **gnn_kwargs,
    ):
        super().__init__()
        self.k_max = k_max
        self.critic_aggregation = critic_aggregation

        if backbone == "dummy":
            self.enc_actor = _DummyEgoEncoder(in_dim, hidden)
            self.enc_critic = _DummyEgoEncoder(in_dim, hidden)
        elif backbone == "sage":
            self.enc_actor = _SAGEEgoEncoder(in_dim, hidden, **gnn_kwargs)
            self.enc_critic = _SAGEEgoEncoder(in_dim, hidden, **gnn_kwargs)
        else:
            raise ValueError(f"Unknown backbone '{backbone}'")


        self.actor_norm = nn.LayerNorm(hidden)
        # Actor: per-node score
        self.actor_head = nn.Linear(hidden, 1)

        # Critic: head maps an H-dim vector to 1 scalar
        self.critic_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        # If using attention aggregation for the joint critic, learn weights over robots
        if self.critic_aggregation == "joint_attn":
            self.attn_score = nn.Linear(hidden, 1)  # produces unnormalized weights per robot

    @torch.no_grad()
    def _remap_indices(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Map padded indices 0..N_max-1 -> local 0..n_i-1; -1 for masked-out.
        mask: [N_max] bool
        """
        N_max = mask.numel()
        idx_map = torch.full((N_max,), -1, dtype=torch.long, device=mask.device)
        count = int(mask.sum().item())              # cast to int for type checkers
        idx_map[mask] = torch.arange(count, device=mask.device)
        return idx_map

    def _build_graph_lists(self, obs):
        """
        Convert padded batch into per-graph lists and candidate indices (remapped).
        Returns:
            x_list, ei_list, cand_loc_idx (list of LongTensor [k_i]), R
        """
        x = obs["x"]                    # [R, N_max, F]
        node_mask = obs["node_mask"]    # [R, N_max] bool
        R, N_max, _ = x.shape

        has_edges = ("edge_index" in obs) and ("edge_mask" in obs)

        x_list: List[torch.Tensor] = []
        ei_list: List[torch.Tensor] = []
        cand_loc_idx: List[torch.Tensor] = []

        for i in range(R):
            mask_i = node_mask[i].to(dtype=torch.bool)      # [N_max] bool
            x_i_full = x[i].to(dtype=torch.float32)         # [N_max, F] float32

            # If no valid nodes, create a 1-node dummy graph to keep encoders happy
            if mask_i.any():
                x_i = x_i_full[mask_i]                      # [n_i, F] 
            else:
                x_i = torch.zeros((1, x.size(-1)),device=x.device, dtype=torch.float32)
                # create a synthetic mask with a single True at position 0 so idx_map is consistent
                mask_i = torch.zeros_like(mask_i)
                mask_i[0] = True

            # map 0..N_max-1 -> -1 or 0..n_i-1
            idx_map = self._remap_indices(mask_i)           # [N_max] -> [-1 or 0..n_i-1] long

            # ---- edges: mask -> cast -> remap -> drop invalid ----
            if has_edges:
                ei_full = obs["edge_index"][i].to(dtype=torch.long)     # [2, E_max]
                emask_i = obs["edge_mask"][i].to(dtype=torch.bool)      # [E_max]
                ei_i = ei_full[:, emask_i]                              # [2, e_i]
                ei_i = idx_map[ei_i]                         # remap to local ids (long)
                valid = (ei_i >= 0).all(dim=0)               # drop edges touching masked nodes
                ei_i = ei_i[:, valid]
            else:
                ei_i = x_i.new_zeros((2, 0), dtype=torch.long)

            # ---- candidates: mask -> cast -> remap -> drop invalid ----
            cand_full = obs["cand_idx"][i].to(dtype=torch.long)          # [K_max]
            cmask_i = obs["cand_mask"][i].to(dtype=torch.bool)           # [K_max]
            cand_raw = cand_full[cmask_i]                                 # [k_i_raw]
            cand_i = idx_map[cand_raw]                                    # -> 0..n_i-1 or -1
            cand_i = cand_i[cand_i >= 0]                                  # keep only valid                                  # keep only valid

            x_list.append(x_i)
            ei_list.append(ei_i)
            cand_loc_idx.append(cand_i)

        return x_list, ei_list, cand_loc_idx, R


    def forward(self, obs):
        x_list, ei_list, cand_loc_idx, R = self._build_graph_lists(obs)
        K_max = obs["cand_mask"].shape[1]
        device = obs["x"].device

        # === Actor (unchanged) ===
        #h_a, batch_a = self.enc_actor.encode_graphs(x_list, ei_list)  # [sum_nodes, H]
        #scores = self.actor_head(h_a).squeeze(-1)                      # [sum_nodes]

         # === Actor (with normalization) ===
        h_a, batch_a = self.enc_actor.encode_graphs(x_list, ei_list)   # [sum_nodes, H]
        h_a = self.actor_norm(h_a)                                     # NEW
        scores = self.actor_head(h_a).squeeze(-1)                      # [sum_nodes]

        logits_list: List[torch.Tensor] = []
        for i, _x_i in enumerate(x_list):
            mask_i = (batch_a.batch == i)
            scores_i = scores[mask_i]                 # [n_i]
            cand_i = cand_loc_idx[i]                  # [k_i]
            
            if cand_i.numel() > 0 and scores_i.numel() > 0:
                raw_logits = scores_i[cand_i]
                li = torch.tanh(raw_logits) * 5.0         # logits in [-5, +5]
            else:
                li = torch.empty(0, device=device)


            if li.numel() < K_max:
                li = torch.cat([li, torch.full((K_max - li.numel(),), -1e9, device=device)], dim=0)
            logits_list.append(li)
        logits = torch.stack(logits_list, dim=0)      # [R, K_max]

        # === Critic (aggregation options) ===
        h_c, batch_c = self.enc_critic.encode_graphs(x_list, ei_list)  # [sum_nodes, H]
        # First, get one embedding per robot by mean-pooling its nodes
        robot_embeds: List[torch.Tensor] = []
        for i, _x_i in enumerate(x_list):
            mask_i = (batch_c.batch == i)
            if mask_i.any():
                h_i = h_c[mask_i].mean(dim=0, keepdim=True)   # [1, H]
            else:
                h_i = torch.zeros(1, h_c.size(-1), device=device)
            robot_embeds.append(h_i)
        E = torch.cat(robot_embeds, dim=0)  # [R, H]

        if self.critic_aggregation == "per_robot":
            # Value per robot: [R]
            v = self.critic_head(E).squeeze(-1)  # [R,1] -> [R]
            return logits, v

        elif self.critic_aggregation == "joint_mean":
            # Simple permutation-invariant pooling: mean over robots -> scalar
            g = E.mean(dim=0, keepdim=True)      # [1, H]
            v = self.critic_head(g).squeeze()    # scalar (0-dim tensor)
            return logits, v

        elif self.critic_aggregation == "joint_attn":
            # Learned attention over robots (still permutation-invariant)
            # scores: [R,1] -> weights: [R,1] via softmax
            a = self.attn_score(E)                        # [R, 1]
            w = torch.softmax(a.squeeze(-1), dim=0).unsqueeze(-1)  # [R,1]
            g = (w * E).sum(dim=0, keepdim=True)          # weighted sum -> [1, H]
            v = self.critic_head(g).squeeze()             # scalar
            return logits, v

        else:
            raise ValueError(f"Unknown critic_aggregation '{self.critic_aggregation}'")
