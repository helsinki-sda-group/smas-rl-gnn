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
        use_two_hop_actor: bool = False,
        use_two_hop_critic: bool = False,
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
        self.use_two_hop_actor = bool(use_two_hop_actor)
        self.use_two_hop_critic = bool(use_two_hop_critic)
        self.eta_index = int(eta_index)
        self._comp_log_sums: Optional[torch.Tensor] = None
        self._comp_log_count: int = 0
        self._comp_log_comp_count: int = 0
        self._comp_log_robot_count: int = 0

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
            self.phi_comp = nn.Sequential(
                nn.Linear(hidden + in_dim + edge_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.score_comp = nn.Linear(hidden, 1, bias=False)
            self.comp_head = nn.Linear(hidden, 1, bias=False)
            self.comp_bias = nn.Parameter(torch.tensor(0.0))
            nn.init.zeros_(self.comp_head.weight)

        # Critic: head maps an H-dim vector to 1 scalar
        self.critic_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        # If using attention aggregation for the joint critic, learn weights over robots
        if self.critic_aggregation == "joint_attn":
            self.attn_score = nn.Linear(hidden, 1)  # produces unnormalized weights per robot

    def _init_comp_stats(self, device: torch.device) -> None:
        if self._comp_log_sums is None:
            self._comp_log_sums = torch.zeros((21,), device=device)
            self._comp_log_count = 0
            self._comp_log_comp_count = 0
            self._comp_log_robot_count = 0

    def pop_comp_norm_stats(self) -> Optional[dict[str, float]]:
        if self._comp_log_sums is None or self._comp_log_count <= 0:
            return None
        sums = self._comp_log_sums.detach().cpu().numpy().astype(float)
        count = float(self._comp_log_count)
        comp_count = float(self._comp_log_comp_count)
        robot_count = float(self._comp_log_robot_count)
        if comp_count <= 0:
            comp_count = 1.0
        if robot_count <= 0:
            robot_count = 1.0
        stats = {
            "norm_h": sums[0] / count,
            "norm_z": sums[1] / count,
            "p_has_comp": sums[2] / count,
            "logit_base": sums[3] / count,
            "logit_comp": sums[4] / count,
            "logit_ind": sums[5] / count,
            "bias_base": sums[6] / count,
            "norm_w_h": sums[7] / count,
            "norm_w_c": sums[8] / count,
            "norm_w_s": sums[9] / count,
            "norm_w_d": sums[10] / count,
            "attn_entropy": sums[11] / comp_count,
            "max_attn": sums[12] / comp_count,
            "ratio_comp_base": sums[13] / robot_count,
            "ratio_comp_gap": sums[14] / robot_count,
            "norm_u": sums[15] / comp_count,
            "std_comp": sums[16] / robot_count,
            "mean_num_comp": sums[17] / comp_count,
            "max_num_comp": sums[18] / robot_count,
            "mean_score": sums[19] / comp_count,
            "std_score": sums[20] / comp_count,
            "count": int(self._comp_log_count),
        }
        self._comp_log_sums = None
        self._comp_log_count = 0
        self._comp_log_comp_count = 0
        self._comp_log_robot_count = 0
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
        h_t: torch.Tensor,
        x_full_i: torch.Tensor,
        ei_full_i: torch.Tensor,
        ea_full_i: torch.Tensor,
        cand_full_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Architecture 3.2: attention over competitors.
        Returns z_t_comp [k_i, H] and indicator [k_i] for whether competitors exist.
        """
        device = x_full_i.device
        k_i = int(cand_full_i.numel())
        if (not self.use_competitor_fusion) or k_i == 0:
            empty = x_full_i.new_zeros((k_i,), dtype=torch.float32)
            return (
                x_full_i.new_zeros((k_i, self.hidden), dtype=torch.float32),
                empty,
                {
                    "attn_entropy": empty,
                    "max_attn": empty,
                    "norm_u": empty,
                    "mean_score": empty,
                    "std_score": empty,
                    "num_comp": empty,
                },
            )

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

        z_list: List[torch.Tensor] = []
        ind_list: List[torch.Tensor] = []
        ent_list: List[torch.Tensor] = []
        max_list: List[torch.Tensor] = []
        norm_u_list: List[torch.Tensor] = []
        mean_score_list: List[torch.Tensor] = []
        std_score_list: List[torch.Tensor] = []
        num_comp_list: List[torch.Tensor] = []
        for idx, t in enumerate(cand_full_i.tolist()):
            t = int(t)
            comps = []
            if src.numel() > 0:
                comp_ids = dst[src == t]
                comps = [int(n) for n in comp_ids.tolist() if int(n) != 0 and int(n) not in task_set_full]

            if len(comps) == 0:
                z_list.append(h_t.new_zeros((self.hidden,), dtype=torch.float32))
                ind_list.append(h_t.new_tensor(0.0))
                ent_list.append(h_t.new_tensor(0.0))
                max_list.append(h_t.new_tensor(0.0))
                norm_u_list.append(h_t.new_tensor(0.0))
                mean_score_list.append(h_t.new_tensor(0.0))
                std_score_list.append(h_t.new_tensor(0.0))
                num_comp_list.append(h_t.new_tensor(0.0))
                continue

            h_t_i = h_t[idx].unsqueeze(0).expand(len(comps), -1)
            x_c = x_full_i[comps]
            if self.edge_dim > 0:
                a_tc = torch.stack([_edge_attr_between(t, c) for c in comps], dim=0)
            else:
                a_tc = x_full_i.new_zeros((len(comps), 0), dtype=torch.float32)

            u_tc = self.phi_comp(torch.cat([h_t_i, x_c, a_tc], dim=-1))
            s_tc = self.score_comp(u_tc).squeeze(-1)
            alpha = torch.softmax(s_tc, dim=0)
            z_t = (alpha.unsqueeze(-1) * u_tc).sum(dim=0)
            z_list.append(z_t)
            ind_list.append(h_t.new_tensor(1.0))
            ent = -(alpha * (alpha + 1e-12).log()).sum()
            ent_list.append(ent)
            max_list.append(alpha.max())
            norm_u_list.append(u_tc.norm(dim=-1).mean())
            mean_score_list.append(s_tc.mean())
            std_score_list.append(s_tc.std(unbiased=False))
            num_comp_list.append(h_t.new_tensor(float(len(comps))))

        return (
            torch.stack(z_list, dim=0),
            torch.stack(ind_list, dim=0),
            {
                "attn_entropy": torch.stack(ent_list, dim=0),
                "max_attn": torch.stack(max_list, dim=0),
                "norm_u": torch.stack(norm_u_list, dim=0),
                "mean_score": torch.stack(mean_score_list, dim=0),
                "std_score": torch.stack(std_score_list, dim=0),
                "num_comp": torch.stack(num_comp_list, dim=0),
            },
        )

    def forward(self, obs):
        (
            x_list, ei_list, edge_attr_list, cand_loc_idx,
            x_list_full, ei_list_full, edge_attr_list_full, cand_loc_idx_full,
            R,
        ) = self._build_graph_lists(obs)
        K_max = obs["cand_mask"].shape[1]
        device = obs["x"].device

        actor_x_list = x_list_full if self.use_two_hop_actor else x_list
        actor_ei_list = ei_list_full if self.use_two_hop_actor else ei_list
        actor_ea_list = edge_attr_list_full if self.use_two_hop_actor else edge_attr_list
        actor_cand_idx = cand_loc_idx_full if self.use_two_hop_actor else cand_loc_idx

        # === Actor graph ===
        h_a, batch_a = self.enc_actor.encode_graphs(actor_x_list, actor_ei_list, actor_ea_list)
        h_a = self.actor_norm(h_a)

        logits_list: List[torch.Tensor] = []
        for i in range(R):
            mask_i = (batch_a.batch == i)
            h_i = h_a[mask_i]                          # [n_i, H]
            cand_i = actor_cand_idx[i]                # [k_i]

            if cand_i.numel() > 0 and h_i.numel() > 0:
                h_t = h_i[cand_i]                     # [k_i, H]
                base_logits = self.actor_head(h_t).squeeze(-1)
                if self.use_competitor_fusion:
                    z_t, ind_t, comp_stats = self._compute_comp_context(
                        h_t,
                        x_list_full[i],
                        ei_list_full[i],
                        edge_attr_list_full[i],
                        cand_loc_idx_full[i],
                    )
                    if z_t.size(0) != h_t.size(0):
                        raise RuntimeError(
                            f"Competitor-correction size mismatch: z_t={z_t.size(0)} vs cand={h_t.size(0)}"
                        )
                    comp_logits = self.comp_head(z_t).squeeze(-1)
                    ind_logits = self.comp_bias * ind_t
                    raw_logits = base_logits + comp_logits + ind_logits

                    with torch.no_grad():
                        self._init_comp_stats(device)
                        if self._comp_log_sums is not None:
                            h_norm = h_t.detach().norm(dim=-1)
                            z_norm = z_t.detach().norm(dim=-1)
                            ind_det = ind_t.detach()
                            count = float(h_norm.numel())
                            self._comp_log_sums[0] += h_norm.sum().detach()
                            self._comp_log_sums[1] += z_norm.sum().detach()
                            self._comp_log_sums[2] += ind_det.sum().detach()
                            self._comp_log_sums[3] += base_logits.detach().sum().detach()
                            self._comp_log_sums[4] += comp_logits.detach().sum().detach()
                            self._comp_log_sums[5] += ind_logits.detach().sum().detach()
                            self._comp_log_sums[6] += float(self.actor_head.bias.detach()) * count
                            self._comp_log_sums[7] += self.actor_head.weight.detach().norm() * count
                            self._comp_log_sums[8] += self.comp_head.weight.detach().norm() * count
                            self._comp_log_sums[9] += self.score_comp.weight.detach().norm() * count
                            self._comp_log_sums[10] += self.comp_bias.detach().abs() * count

                            comp_mask = (ind_det > 0)
                            if comp_mask.any():
                                ent = comp_stats["attn_entropy"].detach()[comp_mask]
                                max_a = comp_stats["max_attn"].detach()[comp_mask]
                                norm_u = comp_stats["norm_u"].detach()[comp_mask]
                                mean_score = comp_stats["mean_score"].detach()[comp_mask]
                                std_score = comp_stats["std_score"].detach()[comp_mask]
                                num_comp = comp_stats["num_comp"].detach()[comp_mask]
                                comp_n = float(comp_mask.sum().item())
                                self._comp_log_sums[11] += ent.sum().detach()
                                self._comp_log_sums[12] += max_a.sum().detach()
                                self._comp_log_sums[15] += norm_u.sum().detach()
                                self._comp_log_sums[17] += num_comp.sum().detach()
                                self._comp_log_sums[19] += mean_score.sum().detach()
                                self._comp_log_sums[20] += std_score.sum().detach()
                                self._comp_log_comp_count += int(comp_n)

                            base_abs = base_logits.detach().abs().mean()
                            comp_abs = comp_logits.detach().abs().mean()
                            ratio_base = comp_abs / (base_abs + 1e-8)
                            self._comp_log_sums[13] += ratio_base.detach()

                            if base_logits.numel() >= 2:
                                top2 = torch.topk(base_logits.detach(), k=2, largest=True).values
                                gap = (top2[0] - top2[1]).abs()
                                ratio_gap = comp_abs / (gap + 1e-8)
                                self._comp_log_sums[14] += ratio_gap.detach()
                                self._comp_log_sums[16] += comp_logits.detach().std(unbiased=False)
                                self._comp_log_sums[18] += comp_stats["num_comp"].detach().max()
                                self._comp_log_robot_count += 1
                            else:
                                self._comp_log_robot_count += 1

                            self._comp_log_count += int(count)
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

        critic_x_list = x_list_full if self.use_two_hop_critic else x_list
        critic_ei_list = ei_list_full if self.use_two_hop_critic else ei_list
        critic_ea_list = edge_attr_list_full if self.use_two_hop_critic else edge_attr_list

        # === Critic graph ===
        h_c, batch_c = self.enc_critic.encode_graphs(critic_x_list, critic_ei_list, critic_ea_list)
        # First, get one embedding per robot by mean-pooling its nodes
        robot_embeds: List[torch.Tensor] = []
        for i in range(R):
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
