import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


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
    def __init__(self, input_dim, gcn_dim, hidden_dim, nhead, num_layers, num_nodes, forecast_dim=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_gcn = gcn_dim > 0

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if self.use_gcn:
            self.spatial_encoder = GCNSpatialEncoder(gcn_dim, hidden_dim)
        else:
            self.register_buffer('dummy_spatial', torch.zeros(1))  # placeholder

        self.time_pos_embedding = nn.Parameter(torch.randn(1000, hidden_dim))  # supports up to 1000 timesteps

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, dropout=0.2,  nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_dim)
        )

        # self.decoder = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(64, forecast_dim)
        # )

    def forward(self, x_seq, edge_index, edge_weight=None, node_features=None):
        # x_seq: [B, T, N, F]
        B, T, N, F = x_seq.shape
        assert N == self.num_nodes, "Mismatch in number of nodes"

        # Step 1: Project input features
        x_proj = self.input_proj(x_seq)  # [B, T, N, D]

        # Step 2: Spatial encoding via GCN (if enabled)
        if self.use_gcn:
            assert node_features is not None, "Static node features must be provided when gcn_dim > 0"
            spatial_repr = self.spatial_encoder(node_features, edge_index, edge_weight)  # [N, D]
            spatial_repr = spatial_repr.unsqueeze(0).unsqueeze(1).repeat(B, T, 1, 1)  # [B, T, N, D]
        else:
            spatial_repr = torch.zeros(B, T, N, self.hidden_dim, device=x_seq.device)  # no spatial context

        # Step 3: Add time + spatial encoding
        time_pos = self.time_pos_embedding[:T].unsqueeze(1).repeat(1, N, 1)  # [T, N, D]
        time_pos = time_pos.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, T, N, D]

        x = x_proj + spatial_repr + time_pos  # [B, T, N, D]

        # Step 4: Flatten temporal and spatial tokens
        x = x.view(B, T * N, self.hidden_dim)  # [B, T*N, D]

        seq_len = T * N
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        # Step 5: Transformer
        x = self.transformer(x, mask = mask)  # [B, T*N, D]

        # Step 6: Reshape to [B, T, N, D] and take last timestep
        x = x.view(B, T, N, self.hidden_dim)
        x_last = x[:, -1, :, :]  # [B, N, D]

        # Step 7: Decode
        out = self.decoder(x_last)  # [B, N, forecast_dim]
        return out





