import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, out_channels)
        self.gcn2 = GCNConv(out_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.relu(self.gcn1(x, edge_index, edge_weight))
        x = self.relu(self.gcn2(x, edge_index, edge_weight))
        return x  # [N, D]


class TemporalTransformer(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, nhead=4, num_layers=1):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: [B, T, N, F]
        B, T, N, F = x.size()
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)  # [B*N, T, F]
        x = self.proj(x)
        x = self.transformer(x)  # [B*N, T, D]
        x = x[:, -1, :]  # Use last time step
        return x.reshape(B, N, self.hidden_dim)  # [B, N, D]


class FusionAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, temporal_feat, spatial_feat):
        # Both [B, N, D]
        q = self.query(temporal_feat)
        k = self.key(spatial_feat)
        v = self.value(spatial_feat)

        attn_weights = self.softmax((q * k).sum(-1, keepdim=True))  # [B, N, 1]
        fused = temporal_feat + attn_weights * v
        return fused  # [B, N, D]


class SpatioTemporalFusionNet(nn.Module):
    def __init__(self, in_channels, graph_feat_dim, trans_hidden=64, out_channels=1, num_nodes=None):
        super().__init__()

        assert num_nodes is not None, "You must provide num_nodes to use node embedding."
        self.node_embedding = nn.Embedding(num_nodes, graph_feat_dim)

        self.temporal_encoder = TemporalTransformer(in_channels, hidden_dim=trans_hidden)
        self.spatial_encoder = GCNEncoder(graph_feat_dim, trans_hidden)
        self.fusion = FusionAttention(trans_hidden)
        self.decoder = nn.Sequential(
            nn.Linear(trans_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )

    def forward(self, x_seq, edge_index, edge_weight=None):
        """
        x_seq: [B, T, N, F] – time series input
        node_features: [N, Fg] – static graph/node input (e.g., centrality, population)
        edge_index: graph edge index
        edge_weight: optional edge weights
        device: device itself
        """
        B, T, N, F = x_seq.size()
        node_features = self.node_embedding(torch.arange(N, device=x_seq.device))

        temporal_repr = self.temporal_encoder(x_seq)      # [B, N, D]
        spatial_repr = self.spatial_encoder(node_features, edge_index, edge_weight)  # [N, D]
        spatial_repr = spatial_repr.unsqueeze(0).repeat(B, 1, 1)  # [B, N, D]

        fused = self.fusion(temporal_repr, spatial_repr)  # [B, N, D]
        output = self.decoder(fused)  # [B, N, out_channels]
        return output
