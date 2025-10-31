import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv

class GCN_GAT_SpatialEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, gat_heads=2):
        super().__init__()
        self.gcn = GCNConv(in_channels, hidden_channels)
        self.gat = GATConv(hidden_channels, hidden_channels, heads=gat_heads, concat=False)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.relu(self.gcn(x, edge_index, edge_weight))
        x = self.relu(self.gat(x, edge_index))
        return x  # [N, hidden_channels]


class SpatiotemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim,              # e.g., 1 ("newCases")
        gcn_dim,                # 0 if no static node feats; else dim of node_features
        hidden_dim,             # Transformer d_model
        nhead,                  # must divide hidden_dim
        num_layers,
        num_nodes,
        forecast_dim=1,
        time_pos_len=1000,
        attn_dropout=0.0,
        ff_dropout=0.2,
        gat_heads=2
    ):
        super().__init__()
        assert hidden_dim % nhead == 0, "hidden_dim must be divisible by nhead"
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_gcn = gcn_dim > 0

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if self.use_gcn:
            self.spatial_encoder = GCN_GAT_SpatialEncoder(gcn_dim, hidden_dim, gat_heads=gat_heads)
        else:
            self.register_buffer('dummy_spatial', torch.zeros(1))

        self.time_pos_embedding = nn.Parameter(torch.randn(time_pos_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=ff_dropout,
            batch_first=True
        )
        # control attention dropout via module patch
        if attn_dropout > 0:
            encoder_layer.self_attn.dropout.p = attn_dropout

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ⬇️ Make decoder depend on hidden_dim so we can sweep it
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, forecast_dim)
        )

    def forward(self, x_seq, edge_index, edge_weight=None, node_features=None):
        # x_seq: [B, T, N, F]
        B, T, N, F = x_seq.shape
        assert N == self.num_nodes, "Mismatch in number of nodes"

        x_proj = self.input_proj(x_seq)  # [B, T, N, D]

        if self.use_gcn:
            assert node_features is not None, "Provide static node features when gcn_dim > 0"
            spatial_repr = self.spatial_encoder(node_features, edge_index, edge_weight)  # [N, D]
            spatial_repr = spatial_repr.unsqueeze(0).unsqueeze(1).repeat(B, T, 1, 1)
        else:
            spatial_repr = torch.zeros(B, T, N, self.hidden_dim, device=x_seq.device)

        time_pos = self.time_pos_embedding[:T].unsqueeze(1).repeat(1, N, 1)  # [T, N, D]
        time_pos = time_pos.unsqueeze(0).repeat(B, 1, 1, 1)                  # [B, T, N, D]

        x = x_proj + spatial_repr + time_pos                                # [B, T, N, D]
        x = x.view(B, T * N, self.hidden_dim)                               # [B, T*N, D]

        seq_len = T * N
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        x = self.transformer(x, mask=mask)                                  # [B, T*N, D]
        x = x.view(B, T, N, self.hidden_dim)
        x_last = x[:, -1, :, :]                                             # [B, N, D]
        out = self.decoder(x_last)                                          # [B, N, forecast_dim]
        return out
