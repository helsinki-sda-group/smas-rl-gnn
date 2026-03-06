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
    def __init__(self, in_dim: int, hidden: int, edge_dim: int = 0, **gnn_kwargs):
        super().__init__()
        self.impl = EgoGraphEncoder(in_dim, hidden, edge_dim=edge_dim, **gnn_kwargs)

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
        edge_dim: int = 0,
        use_competitor_fusion: bool = True,
        lambda_init: float = 0.0,
        eta_index: int = -1,
        **gnn_kwargs,
    ):
        super().__init__()
        self.k_max = k_max
        self.hidden = hidden
        self.edge_dim = int(edge_dim)
        self.critic_aggregation = critic_aggregation
        self.use_competitor_fusion = bool(use_competitor_fusion)
        self.eta_index = int(eta_index)
        self._comp_norm_sums: Optional[torch.Tensor] = None
        self._comp_norm_count: int = 0

        if backbone == "dummy":
            self.enc_actor = _DummyEgoEncoder(in_dim, hidden)
            self.enc_critic = _DummyEgoEncoder(in_dim, hidden)
        elif backbone == "sage":
            self.enc_actor = _SAGEEgoEncoder(in_dim, hidden, edge_dim=edge_dim, **gnn_kwargs)
            self.enc_critic = _SAGEEgoEncoder(in_dim, hidden, edge_dim=edge_dim, **gnn_kwargs)
        else:
            raise ValueError(f"Unknown backbone '{backbone}'")


        self.actor_norm = nn.LayerNorm(hidden)
        self.actor_head = nn.Linear(hidden, 1)
        if self.use_competitor_fusion:
            self.lambda_comp = nn.Parameter(torch.tensor(lambda_init))

        # Critic: head maps an H-dim vector to 1 scalar
        self.critic_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        # If using attention aggregation for the joint critic, learn weights over robots
        if self.critic_aggregation == "joint_attn":
            self.attn_score = nn.Linear(hidden, 1)  # produces unnormalized weights per robot

    def _init_comp_stats(self, device: torch.device) -> None:
        if self._comp_norm_sums is None:
            self._comp_norm_sums = torch.zeros((4,), device=device)
            self._comp_norm_count = 0

    def pop_comp_norm_stats(self) -> Optional[dict[str, float]]:
        if self._comp_norm_sums is None or self._comp_norm_count <= 0:
            return None
        sums = self._comp_norm_sums.detach().cpu().numpy().astype(float)
        count = float(self._comp_norm_count)
        stats = {
            "h_k": sums[0] / count,
            "m_comp": sums[1] / count,
            "lam_m_comp": sums[2] / count,
            "lam": sums[3] / count,
            "count": int(self._comp_norm_count),
        }
        self._comp_norm_sums = None
        self._comp_norm_count = 0
        return stats

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
        Build both:
        (a) full remapped graphs (ego + tasks + competitors)
        (b) 1-hop pruned graphs (ego + candidate tasks)
        """
        x = obs["x"]                    # [R, N_max, F]
        node_mask = obs["node_mask"]    # [R, N_max] bool
        R, _, _ = x.shape

        has_edges = ("edge_index" in obs) and ("edge_mask" in obs)
        has_edge_attr = has_edges and ("edge_attr" in obs)

        x_list_full: List[torch.Tensor] = []
        ei_list_full: List[torch.Tensor] = []
        edge_attr_list_full: List[torch.Tensor] = []
        cand_loc_idx_full: List[torch.Tensor] = []

        x_list_1hop: List[torch.Tensor] = []
        ei_list_1hop: List[torch.Tensor] = []
        edge_attr_list_1hop: List[torch.Tensor] = []
        cand_loc_idx_1hop: List[torch.Tensor] = []

        for i in range(R):
            mask_i = node_mask[i].to(dtype=torch.bool)
            x_i_padded = x[i].to(dtype=torch.float32)

            if mask_i.any():
                x_full = x_i_padded[mask_i]
                full_mask = mask_i.clone()
            else:
                x_full = torch.zeros((1, x.size(-1)), device=x.device, dtype=torch.float32)
                full_mask = torch.zeros_like(mask_i)
                full_mask[0] = True

            idx_map_full = self._remap_indices(full_mask)

            if has_edges:
                ei_full_pad = obs["edge_index"][i].to(dtype=torch.long)
                emask_i = obs["edge_mask"][i].to(dtype=torch.bool)
                ei_full = ei_full_pad[:, emask_i]
                ei_full = idx_map_full[ei_full]
                valid_full = (ei_full >= 0).all(dim=0)
                ei_full = ei_full[:, valid_full]

                if has_edge_attr:
                    ea_full_pad = obs["edge_attr"][i].to(dtype=torch.float32)
                    ea_full = ea_full_pad[emask_i][valid_full]
                else:
                    ea_full = x_full.new_zeros((ei_full.size(1), 0), dtype=torch.float32)
            else:
                ei_full = x_full.new_zeros((2, 0), dtype=torch.long)
                ea_full = x_full.new_zeros((0, 0), dtype=torch.float32)

            cand_full_pad = obs["cand_idx"][i].to(dtype=torch.long)
            cmask_i = obs["cand_mask"][i].to(dtype=torch.bool)
            cand_raw = cand_full_pad[cmask_i]
            cand_full = idx_map_full[cand_raw]
            valid_cand = (cand_full >= 0)
            cand_raw_valid = cand_raw[valid_cand]
            cand_full = cand_full[valid_cand]

            x_list_full.append(x_full)
            ei_list_full.append(ei_full)
            edge_attr_list_full.append(ea_full)
            cand_loc_idx_full.append(cand_full)

            keep_mask = torch.zeros_like(full_mask)
            if full_mask[0]:
                keep_mask[0] = True
            if cand_raw_valid.numel() > 0:
                keep_mask[cand_raw_valid] = True

            if keep_mask.any():
                x_1hop = x_i_padded[keep_mask]
            else:
                x_1hop = torch.zeros((1, x.size(-1)), device=x.device, dtype=torch.float32)
                keep_mask = torch.zeros_like(keep_mask)
                keep_mask[0] = True

            idx_map_1hop = self._remap_indices(keep_mask)

            if has_edges:
                ei_1hop_pad = obs["edge_index"][i].to(dtype=torch.long)
                emask_i = obs["edge_mask"][i].to(dtype=torch.bool)
                ei_1hop = ei_1hop_pad[:, emask_i]
                ei_1hop = idx_map_1hop[ei_1hop]
                valid_1hop = (ei_1hop >= 0).all(dim=0)
                ei_1hop = ei_1hop[:, valid_1hop]

                if has_edge_attr:
                    ea_1hop_pad = obs["edge_attr"][i].to(dtype=torch.float32)
                    ea_1hop = ea_1hop_pad[emask_i][valid_1hop]
                else:
                    ea_1hop = x_1hop.new_zeros((ei_1hop.size(1), 0), dtype=torch.float32)
            else:
                ei_1hop = x_1hop.new_zeros((2, 0), dtype=torch.long)
                ea_1hop = x_1hop.new_zeros((0, 0), dtype=torch.float32)

            cand_1hop = idx_map_1hop[cand_raw_valid]
            cand_1hop = cand_1hop[cand_1hop >= 0]

            x_list_1hop.append(x_1hop)
            ei_list_1hop.append(ei_1hop)
            edge_attr_list_1hop.append(ea_1hop)
            cand_loc_idx_1hop.append(cand_1hop)

        return (
            x_list_1hop, ei_list_1hop, edge_attr_list_1hop, cand_loc_idx_1hop,
            x_list_full, ei_list_full, edge_attr_list_full, cand_loc_idx_full,
            R,
        )


    def _compute_comp_context(
        self,
        x_full_i: torch.Tensor,
        ei_full_i: torch.Tensor,
        ea_full_i: torch.Tensor,
        cand_full_i: torch.Tensor,
    ) -> torch.Tensor:
        """
        Architecture 3.1: s_t = min_{r in C(t)} ETA(r,t) - ETA(ego,t)
        Returns s_t in the same order as cand_full_i. Shape [k_i].
        """
        device = x_full_i.device
        k_i = int(cand_full_i.numel())
        if (not self.use_competitor_fusion) or k_i == 0:
            return x_full_i.new_zeros((k_i,), dtype=torch.float32)
        if self.edge_dim <= 0 or self.eta_index < 0:
            return x_full_i.new_zeros((k_i,), dtype=torch.float32)

        src = ei_full_i[0] if ei_full_i.numel() > 0 else torch.empty(0, dtype=torch.long, device=device)
        dst = ei_full_i[1] if ei_full_i.numel() > 0 else torch.empty(0, dtype=torch.long, device=device)
        task_set_full = set(int(t) for t in cand_full_i.tolist())

        edge_attr_map = {}
        if ea_full_i.numel() > 0:
            for k in range(ei_full_i.size(1)):
                u = int(src[k].item())
                v = int(dst[k].item())
                edge_attr_map[(u, v)] = ea_full_i[k]

        def _edge_attr_between(u: int, v: int) -> torch.Tensor:
            out = edge_attr_map.get((u, v), None)
            if out is None:
                out = edge_attr_map.get((v, u), None)
            if out is None:
                out = x_full_i.new_zeros((self.edge_dim,), dtype=torch.float32)
            return out

        s_list: List[torch.Tensor] = []
        for t in cand_full_i.tolist():
            t = int(t)
            eta_ego = _edge_attr_between(0, t)[self.eta_index]
            comps = []
            if src.numel() > 0:
                comp_ids = dst[src == t]
                comps = [int(n) for n in comp_ids.tolist() if int(n) != 0 and int(n) not in task_set_full]

            if len(comps) == 0:
                s_list.append(eta_ego.new_tensor(0.0))
            else:
                eta_comp = torch.stack([_edge_attr_between(c, t)[self.eta_index] for c in comps], dim=0)
                s_list.append(eta_comp.min() - eta_ego)

        return torch.stack(s_list, dim=0)

    def forward(self, obs):
        (
            x_list, ei_list, edge_attr_list, cand_loc_idx,
            x_list_full, ei_list_full, edge_attr_list_full, cand_loc_idx_full,
            R,
        ) = self._build_graph_lists(obs)
        K_max = obs["cand_mask"].shape[1]
        device = obs["x"].device

        # === Actor on 1-hop graph ===
        h_a, batch_a = self.enc_actor.encode_graphs(x_list, ei_list, edge_attr_list)
        h_a = self.actor_norm(h_a)

        logits_list: List[torch.Tensor] = []
        for i, _x_i in enumerate(x_list):
            mask_i = (batch_a.batch == i)
            h_i = h_a[mask_i]                          # [n_i, H]
            cand_i = cand_loc_idx[i]                  # [k_i]

            if cand_i.numel() > 0 and h_i.numel() > 0:
                h_t = h_i[cand_i]                     # [k_i, H]
                base_logits = self.actor_head(h_t).squeeze(-1)
                if self.use_competitor_fusion:
                    s_t = self._compute_comp_context(
                        x_list_full[i],
                        ei_list_full[i],
                        edge_attr_list_full[i],
                        cand_loc_idx_full[i],
                    )
                    if s_t.numel() != h_t.size(0):
                        raise RuntimeError(
                            f"Competitor-correction size mismatch: s_t={s_t.size(0)} vs cand={h_t.size(0)}"
                        )
                    lam = self.lambda_comp
                    raw_logits = base_logits + lam * s_t

                    with torch.no_grad():
                        self._init_comp_stats(device)
                        if self._comp_norm_sums is not None:
                            h_norm = h_t.detach().norm(dim=-1)
                            m_norm = s_t.detach().abs()
                            lam_det = lam.detach()
                            self._comp_norm_sums[0] += h_norm.sum().detach()
                            self._comp_norm_sums[1] += m_norm.sum().detach()
                            self._comp_norm_sums[2] += (lam_det * m_norm).sum().detach()
                            self._comp_norm_sums[3] += lam_det * float(h_norm.numel())
                            self._comp_norm_count += int(h_norm.numel())
                else:
                    raw_logits = base_logits
                li = raw_logits
            else:
                li = torch.empty(0, device=device)

            if li.numel() < K_max:
                li = torch.cat(
                    [li, torch.full((K_max - li.numel(),), -1e9, device=device)],
                    dim=0,
                )
            logits_list.append(li)
        logits = torch.stack(logits_list, dim=0)      # [R, K_max]

        # === Critic on full graph ===
        h_c, batch_c = self.enc_critic.encode_graphs(x_list_full, ei_list_full, edge_attr_list_full)
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
