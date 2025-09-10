import torch
import torch.nn as nn
from typing import List
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops

class DummyBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EgoGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int = 2):
        super().__init__()
        gs = [SAGEConv(in_dim, hidden)]
        for _ in range(layers - 1):
            gs.append(SAGEConv(hidden, hidden))
        self.gnn = nn.ModuleList(gs)
        self.act = nn.ReLU()
        # optional head:
        # self.post = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.1))

    def forward(self, x_list: List[torch.Tensor], edge_index_list: List[torch.Tensor], _batch_list=None):
        """
        Encode a batch of ego-graphs into node embeddings using GraphSAGE.

        Args:
            x_list:  list of [n_i, F] node feature tensors
            edge_index_list: list of [2, e_i] edge index tensors per graph
            _batch_list: unused (kept for signature compatibility)

        Returns:
            h:     [sum_i n_i, hidden] node embeddings
            batch: PyG Batch with .batch mapping nodes -> graph ids
        """
        datas: List[Data] = [Data(x=x, edge_index=ei) for x, ei in zip(x_list, edge_index_list)]
        batch: Batch = Batch.from_data_list(datas) # type: ignore

        # add self-loops once on the merged graph (helps isolated nodes / empty edge sets)
        edge_index, _ = add_self_loops(batch.edge_index, num_nodes=batch.x.size(0)) # type: ignore

        h = batch.x # type: ignore
        for conv in self.gnn:
            h = self.act(conv(h, edge_index))
        # h = self.post(h)
        return h, batch
