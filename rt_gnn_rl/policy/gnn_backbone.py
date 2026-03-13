import torch
import torch.nn as nn
from typing import List, Optional
from torch_geometric.nn import SAGEConv, MessagePassing
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
    def __init__(self, in_dim: int, hidden: int, layers: int = 2, edge_dim: int = 0):
        super().__init__()
        self.edge_dim = int(edge_dim)
        if self.edge_dim > 0:
            gs = [EdgeSAGEConv(in_dim, hidden, self.edge_dim)]
            for _ in range(layers - 1):
                gs.append(EdgeSAGEConv(hidden, hidden, self.edge_dim))
        else:
            gs = [SAGEConv(in_dim, hidden)]
            for _ in range(layers - 1):
                gs.append(SAGEConv(hidden, hidden))
        self.gnn = nn.ModuleList(gs)
        self.act = nn.ReLU()
        # optional head:
        # self.post = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.1))

    def forward(
        self,
        x_list: List[torch.Tensor],
        edge_index_list: List[torch.Tensor],
        edge_attr_list: Optional[List[torch.Tensor]] = None,
        _batch_list=None,
    ):
        """
        Encode a batch of ego-graphs into node embeddings using GraphSAGE.

        Args:
            x_list:  list of [n_i, F] node feature tensors
            edge_index_list: list of [2, e_i] edge index tensors per graph
            _batch_list: unused placeholder that preserves compatibility with the unified encoder API.

        Returns:
            h:     [sum_i n_i, hidden] node embeddings
            batch: PyG Batch with .batch mapping nodes -> graph ids
        """
        datas: List[Data] = []
        if edge_attr_list is None:
            for x, ei in zip(x_list, edge_index_list):
                datas.append(Data(x=x, edge_index=ei))
        else:
            for x, ei, ea in zip(x_list, edge_index_list, edge_attr_list):
                datas.append(Data(x=x, edge_index=ei, edge_attr=ea))
        batch: Batch = Batch.from_data_list(datas) # type: ignore

        # add self-loops once on the merged graph (helps isolated nodes / empty edge sets)
        if self.edge_dim > 0:
            edge_index, edge_attr = add_self_loops(
                batch.edge_index,
                batch.edge_attr,
                num_nodes=batch.x.size(0),
                fill_value=0.0,
            )
        else:
            edge_index, edge_attr = add_self_loops(batch.edge_index, num_nodes=batch.x.size(0)) # type: ignore

        h = batch.x # type: ignore
        for conv in self.gnn:
            if isinstance(conv, EdgeSAGEConv):
                h = self.act(conv(h, edge_index, edge_attr))
            else:
                h = self.act(conv(h, edge_index))
        # h = self.post(h)
        return h, batch


class EdgeSAGEConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int):
        super().__init__(aggr="mean")
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
