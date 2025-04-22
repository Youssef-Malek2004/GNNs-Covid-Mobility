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


class TemporalTransformer(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, nhead=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hidden_dim = hidden_dim

    def forward(self, x_seq):
        # x_seq: [B, T, N, F]
        B, T, N, F = x_seq.size()
        x = x_seq.permute(0, 2, 1, 3).reshape(B * N, T, F)  # [B*N, T, F]
        x = self.proj(x)  # [B*N, T, D]
        x = self.transformer(x)  # [B*N, T, D]
        x = x[:, -1, :]  # last time step
        return x.reshape(B, N, self.hidden_dim)  # [B, N, D]


class GATTemporalFusionNet(nn.Module):
    def __init__(self, in_channels, graph_feat_dim, hidden_dim=64, out_channels=1,
                 num_nodes=None, dropout=0.2, nhead=4, num_layers=1):
        super().__init__()
        assert num_nodes is not None, "You must provide num_nodes."

        self.node_embedding = nn.Embedding(num_nodes, graph_feat_dim)
        self.temporal_encoder = TemporalTransformer(in_channels, hidden_dim, nhead, num_layers, dropout)
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

        # Encode temporal features with Transformer
        temporal_repr = self.temporal_encoder(x_seq)  # [B, N, D]

        # Encode spatial structure via GAT
        node_features = self.node_embedding(torch.arange(N, device=x_seq.device))  # [N, Fg]
        spatial_repr = self.spatial_encoder(node_features, edge_index, edge_weight)  # [N, D]
        spatial_repr = spatial_repr.unsqueeze(0).repeat(B, 1, 1)  # [B, N, D]

        # Fuse temporal + spatial
        fused = torch.cat([temporal_repr, spatial_repr], dim=-1)  # [B, N, 2D]
        fused = self.fusion(fused)
        fused = self.fusion_dropout(fused)

        return self.decoder(fused)  # [B, N, 1]
