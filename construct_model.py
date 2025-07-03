import os
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph as skgraph
from scipy import sparse as sp
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, NNConv, global_max_pool
from torch_scatter import scatter_mean


class EdgeNN(nn.Module):
    """
    Embedding according to edge type, and then modulated by edge features.
    """

    def __init__(self, in_channels, out_channels, device, n_edge_types=25):
        super(EdgeNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.edge_type_embedding = nn.Embedding(n_edge_types, out_channels)
        self.fc_h = nn.Linear(in_channels, out_channels)
        self.fc_g = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_edges, 1(edge_type) + in_channels]
        Return: [batch_size, out_channels]
        """
        y = self.edge_type_embedding(x[..., 0].clone().detach().type(torch.long).to(self.device))
        h = self.fc_h(x[..., 1:(1 + self.in_channels)].clone().detach().type(torch.float).to(self.device))
        g = self.fc_g(x[..., 1:(1 + self.in_channels)].clone().detach().type(torch.float).to(self.device))
        y = y * h + g
        return F.relu(y, inplace=True)
    

class CellNet(nn.Module):

    def __init__(self, in_channels, out_channels, device,
                 batch=True, edge_features=2, n_edge_types=25):
        """
        Args:
            in_channels: number of node features
            out_channels: number of output node features
            batch: True if from DataLoader; False if single Data object
            edge_features: number of edge features (excluding edge type)
            n_edge_types: number of edge types
        """
        super(CellNet, self).__init__()
        self.device = device
        self.batch = batch

        self.conv1 = NNConv(
            in_channels, 10,
            EdgeNN(edge_features, in_channels*10, device, n_edge_types=n_edge_types),
            aggr='mean', root_weight=True, bias=True
        )
        self.conv2 = NNConv(
            10, 10,
            EdgeNN(edge_features, 10*10, device, n_edge_types=n_edge_types),
            aggr='mean', root_weight=True, bias=True
        )
        self.conv3 = NNConv(
            10, out_channels,
            EdgeNN(edge_features, 10*out_channels, device, n_edge_types=n_edge_types),
            aggr='mean', root_weight=True, bias=True
        )

    def forward(self, data):
        """
        Args:
            data: Data in torch_geometric.data
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)

        gate = torch.eq(
            data.cell_type, 1
        ).clone().detach().requires_grad_(False).type(torch.long).to(self.device)
        if self.batch:
            _batch_size = data.batch[-1] + 1
            x = scatter_mean(x, gate * (data.batch+1), dim=0).to(self.device)[1:_batch_size+1, :]
        else:
            x = scatter_mean(x, gate, dim=0).to(self.device)[1, :]

        return x