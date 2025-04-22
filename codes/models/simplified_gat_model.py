import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_channels, out_channels, heads=4, concat=False, edge_dim=1)
        self.gat2 = GATConv(out_channels, out_channels, heads=4, concat=False, edge_dim=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.relu(self.gat1(x, edge_index, edge_attr=edge_weight))
        x = self.dropout(x)
        x = self.relu(self.gat2(x, edge_index, edge_attr=edge_weight))
        x = self.dropout(x)
        return x  # [N, D]


class GATFusionNet(nn.Module):
    def __init__(self, in_channels, graph_feat_dim, hidden_dim=64, out_channels=1, num_nodes=None, dropout=0.2):
        super().__init__()

        assert num_nodes is not None, "You must provide num_nodes to use node embedding."
        self.node_embedding = nn.Embedding(num_nodes, graph_feat_dim)
        self.temporal_proj = nn.Linear(in_channels, hidden_dim)

        self.temporal_dropout = nn.Dropout(dropout)
        self.spatial_encoder = GATEncoder(graph_feat_dim, hidden_dim, dropout=dropout)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_dropout = nn.Dropout(dropout)

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_channels)
        )

    def forward(self, x_seq, edge_index, edge_weight=None):
        B, T, N, F = x_seq.size()

        # Temporal feature projection from last time step
        x_last = x_seq[:, -1, :, :]  # [B, N, F]
        x_last = self.temporal_proj(x_last)  # [B, N, D]
        x_last = self.temporal_dropout(x_last)

        # Static graph-based encoding
        node_features = self.node_embedding(torch.arange(N, device=x_seq.device))  # [N, Fg]
        spatial_repr = self.spatial_encoder(node_features, edge_index, edge_weight)  # [N, D]
        spatial_repr = spatial_repr.unsqueeze(0).repeat(B, 1, 1)  # [B, N, D]

        # Fusion
        fused = torch.cat([x_last, spatial_repr], dim=-1)  # [B, N, 2D]
        fused = self.fusion(fused)  # [B, N, D]
        fused = self.fusion_dropout(fused)

        return self.decoder(fused)  # [B, N, 1]
