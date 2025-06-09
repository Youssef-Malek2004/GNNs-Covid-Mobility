import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def topk_city_selector(node_embeddings: torch.Tensor, k: int) -> torch.LongTensor:
    """
    Args:
        node_embeddings: [N, D] tensor of node features/embeddings
        k: number of top neighbors to select

    Returns:
        neighbor_indices: [N, k] LongTensor with top-k similar cities for each city
    """
    node_norm = F.normalize(node_embeddings, p=2, dim=1)  # [N, D]
    sim_matrix = node_norm @ node_norm.T  # [N, N]
    sim_matrix.fill_diagonal_(-float('inf'))
    topk = torch.topk(sim_matrix, k=k, dim=1)
    return topk.indices  # [N, k]


class GCNSpatialEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, out_channels)
        self.gcn2 = GCNConv(out_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.relu(self.gcn1(x, edge_index, edge_weight))
        x = self.relu(self.gcn2(x, edge_index, edge_weight))
        return x  # [N, D]


class SpatiotemporalTransformer(nn.Module):
    def __init__(self, input_dim, gcn_dim, hidden_dim, nhead, num_layers, num_nodes, k_neighbors=5, forecast_dim=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.k_neighbors = k_neighbors
        self.hidden_dim = hidden_dim
        self.use_gcn = gcn_dim > 0

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if self.use_gcn:
            self.spatial_encoder = GCNSpatialEncoder(gcn_dim, hidden_dim)
        else:
            self.register_buffer('dummy_spatial', torch.zeros(1))

        self.time_pos_embedding = nn.Parameter(torch.randn(1000, hidden_dim))  # for up to 1000 timesteps

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=0.7, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, forecast_dim)
        )

    def forward(self, x_seq, edge_index, edge_weight=None, node_features=None):
        # x_seq: [B, T, N, F]
        B, T, N, F = x_seq.shape
        assert N == self.num_nodes, "Mismatch in number of nodes"

        x_proj = self.input_proj(x_seq)  # [B, T, N, D]

        if self.use_gcn:
            assert node_features is not None
            node_embeddings = self.spatial_encoder(node_features, edge_index, edge_weight)  # [N, D]

            # Select neighbors: shape [N, k]
            topk_neighbors = topk_city_selector(node_embeddings, self.k_neighbors)  # [N, k]
            neighbor_matrix = torch.cat([
                torch.arange(N, device=x_seq.device).unsqueeze(1),  # include self
                topk_neighbors
            ], dim=1)  # [N, k+1]

            # Gather neighborhoods: shape [B, T, N, k+1, D]
            neighborhoods = torch.stack([
                x_proj[:, :, neighbor_matrix[i]] for i in range(N)
            ], dim=2)  # [B, T, N, k+1, D]

            # Add positional encodings
            pos = self.time_pos_embedding[:T].unsqueeze(1).unsqueeze(0)  # [1, T, 1, D]
            neighborhoods += pos.unsqueeze(3)  # broadcast to [B, T, N, k+1, D]

            # Merge dims: [B, T, N, k+1, D] → [B*N, T*(k+1), D]
            neighborhoods = neighborhoods.permute(0, 2, 1, 3, 4).reshape(B * N, T * (self.k_neighbors + 1),
                                                                         self.hidden_dim)

            # Mask: causal
            seq_len = neighborhoods.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x_seq.device), diagonal=1).bool()

            # Transformer forward
            x = self.transformer(neighborhoods, mask=mask)  # [B*N, T*(k+1), D]

            # Reshape back and decode
            x = x.view(B, N, T, self.k_neighbors + 1, self.hidden_dim)
            x_last = x[:, :, -1, -1, :]  # [B, N, D] — last time, target city
            return self.decoder(x_last)  # [B, N, forecast_dim]

