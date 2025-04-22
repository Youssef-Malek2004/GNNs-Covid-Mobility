import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv


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

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, in_channels, gat_hidden, trans_hidden=64, nhead=4, num_layers=1):
        super().__init__()
        self.gat = GATConv(in_channels, gat_hidden, heads=4, concat=False, edge_dim=1)

        self.proj = nn.Linear(gat_hidden, trans_hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=trans_hidden, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hidden_dim = trans_hidden

    def forward(self, x_seq, edge_index, edge_weight=None):
        # x_seq: [B, T, N, F]
        B, T, N, F = x_seq.size()
        x_seq = x_seq.permute(1, 0, 2, 3)  # [T, B, N, F]

        spatially_enriched = []
        for t in range(T):
            x_t = x_seq[t]  # [B, N, F]
            x_t = x_t.reshape(B * N, F)
            # Apply GCN to each batch sample independently
            x_t_list = []
            for b in range(B):
                x_b = x_t[b * N:(b + 1) * N]  # [N, F]
                x_b = self.gat(x_b, edge_index, edge_attr=edge_weight)  # [N, H]
                x_t_list.append(x_b)
            x_t = torch.stack(x_t_list, dim=0)  # [B, N, H]
            spatially_enriched.append(x_t)

        # Stack across time
        x = torch.stack(spatially_enriched, dim=1)  # [B, T, N, H]
        x = self.proj(x)  # [B, T, N, D]

        # Prepare for transformer
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)  # [B*N, T, D]
        x = self.transformer(x)  # [B*N, T, D]
        x = x[:, -1, :]  # last time step
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

        self.temporal_encoder = SpatioTemporalTransformer(in_channels, gat_hidden=32, trans_hidden=trans_hidden)
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

        temporal_repr = self.temporal_encoder(x_seq, edge_index, edge_weight)  # [B, N, D]
        spatial_repr = self.spatial_encoder(node_features, edge_index, edge_weight)  # [N, D]
        spatial_repr = spatial_repr.unsqueeze(0).repeat(B, 1, 1)  # [B, N, D]

        fused = self.fusion(temporal_repr, spatial_repr)  # [B, N, D]
        output = self.decoder(fused)  # [B, N, out_channels]
        return output
