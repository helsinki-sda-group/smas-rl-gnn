import unittest

import torch

from rt_gnn_rl.policy.actor_critic import EgoActorCritic


class TwoHopCompContextTests(unittest.TestCase):
    def _make_obs(self, n_nodes, cand_nodes, edge_pairs, f_dim=6):
        k_max = len(cand_nodes)
        e_max = max(1, len(edge_pairs))

        x = torch.zeros((1, n_nodes, f_dim), dtype=torch.float32)
        for n in range(n_nodes):
            x[0, n, n % f_dim] = 1.0

        node_mask = torch.ones((1, n_nodes), dtype=torch.bool)
        edge_index = torch.zeros((1, 2, e_max), dtype=torch.long)
        edge_mask = torch.zeros((1, e_max), dtype=torch.bool)
        for idx, (u, v) in enumerate(edge_pairs):
            edge_index[0, 0, idx] = int(u)
            edge_index[0, 1, idx] = int(v)
            edge_mask[0, idx] = True

        cand_idx = torch.tensor([cand_nodes], dtype=torch.long)
        cand_mask = torch.ones((1, k_max), dtype=torch.bool)

        return {
            "x": x,
            "node_mask": node_mask,
            "edge_index": edge_index,
            "edge_mask": edge_mask,
            "cand_idx": cand_idx,
            "cand_mask": cand_mask,
        }

    def _run_and_pop_stats(self, model, obs):
        _logits, _value = model.forward(obs)
        return model.pop_comp_norm_stats()

    def test_comp_corr_directed_outgoing_edge_has_nonzero_comp_stats(self):
        model = EgoActorCritic(
            in_dim=6,
            hidden=16,
            k_max=1,
            backbone="dummy",
            edge_dim=0,
            use_competitor_fusion=True,
            comp_fusion_mode="attn",
            use_two_hop_actor=False,
        )
        obs = self._make_obs(
            n_nodes=3,
            cand_nodes=[1],
            edge_pairs=[(0, 1), (1, 0), (1, 2)],
        )

        stats = self._run_and_pop_stats(model, obs)

        self.assertIsNotNone(stats)
        self.assertGreater(stats["p_has_comp"], 0.0)
        self.assertGreater(stats["mean_num_comp"], 0.0)
        self.assertEqual(stats["count"], 1)

    def test_comp_corr_directed_incoming_only_edge_does_not_count_competitor(self):
        model = EgoActorCritic(
            in_dim=6,
            hidden=16,
            k_max=1,
            backbone="dummy",
            edge_dim=0,
            use_competitor_fusion=True,
            comp_fusion_mode="attn",
            use_two_hop_actor=False,
        )
        obs = self._make_obs(
            n_nodes=3,
            cand_nodes=[1],
            edge_pairs=[(0, 1), (1, 0), (2, 1)],
        )

        stats = self._run_and_pop_stats(model, obs)

        self.assertIsNotNone(stats)
        self.assertEqual(stats["p_has_comp"], 0.0)
        self.assertEqual(stats["mean_num_comp"], 0.0)

    def test_comp_corr_directed_mean_num_comp_aggregates_across_candidates(self):
        model = EgoActorCritic(
            in_dim=6,
            hidden=16,
            k_max=2,
            backbone="dummy",
            edge_dim=0,
            use_competitor_fusion=True,
            comp_fusion_mode="attn",
            use_two_hop_actor=False,
        )
        obs = self._make_obs(
            n_nodes=5,
            cand_nodes=[1, 2],
            edge_pairs=[
                (0, 1),
                (1, 0),
                (0, 2),
                (2, 0),
                (1, 3),
                (2, 3),
                (2, 4),
            ],
        )

        stats = self._run_and_pop_stats(model, obs)

        self.assertIsNotNone(stats)
        self.assertAlmostEqual(stats["p_has_comp"], 1.0, places=6)
        self.assertAlmostEqual(stats["mean_num_comp"], 1.5, places=6)
        self.assertEqual(stats["count"], 2)

    def test_comp_corr_maxpool_also_reports_nonzero_comp_stats(self):
        model = EgoActorCritic(
            in_dim=6,
            hidden=16,
            k_max=1,
            backbone="dummy",
            edge_dim=0,
            use_competitor_fusion=True,
            comp_fusion_mode="maxpool",
            use_two_hop_actor=False,
        )
        obs = self._make_obs(
            n_nodes=3,
            cand_nodes=[1],
            edge_pairs=[(0, 1), (1, 0), (1, 2)],
        )

        stats = self._run_and_pop_stats(model, obs)

        self.assertIsNotNone(stats)
        self.assertGreater(stats["p_has_comp"], 0.0)
        self.assertGreater(stats["mean_num_comp"], 0.0)

    def test_two_hop_plain_has_no_comp_norm_stats(self):
        model = EgoActorCritic(
            in_dim=6,
            hidden=16,
            k_max=1,
            backbone="dummy",
            edge_dim=0,
            use_competitor_fusion=False,
            use_two_hop_actor=True,
        )
        obs = self._make_obs(
            n_nodes=3,
            cand_nodes=[1],
            edge_pairs=[(0, 1), (1, 0), (1, 2)],
        )

        stats = self._run_and_pop_stats(model, obs)

        self.assertIsNone(stats)


if __name__ == "__main__":
    unittest.main()
