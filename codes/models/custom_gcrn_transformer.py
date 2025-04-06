import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GConvGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GConvGRUCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.conv_z = GCNConv(in_channels + hidden_channels, hidden_channels)
        self.conv_r = GCNConv(in_channels + hidden_channels, hidden_channels)
        self.conv_h = GCNConv(in_channels + hidden_channels, hidden_channels)

    def forward(self, x, h, edge_index, edge_weight=None):
        # x, h: [B, N, F]
        B, N, F = x.size()
        h_next = []

        for b in range(B):
            xh = torch.cat([x[b], h[b]], dim=-1)
            z = torch.sigmoid(self.conv_z(xh, edge_index, edge_weight))
            r = torch.sigmoid(self.conv_r(xh, edge_index, edge_weight))
            rh = torch.cat([x[b], r * h[b]], dim=-1)
            h_tilde = torch.tanh(self.conv_h(rh, edge_index, edge_weight))
            h_new = (1 - z) * h[b] + z * h_tilde
            h_next.append(h_new)

        return torch.stack(h_next, dim=0)  # [B, N, H]


class GCRNWithTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, transformer_dim=64, nhead=4, num_layers=1):
        super(GCRNWithTransformer, self).__init__()
        self.transformer_input_proj = nn.Linear(in_channels, transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=transformer_dim * 2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_back = nn.Linear(transformer_dim, in_channels)

        self.gconv_gru = GConvGRUCell(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_seq, edge_index, edge_weight=None):
        """
        x_seq: [B, T, N, F]
        Returns: [B, N, out_channels]
        """
        if x_seq.dim() == 3:  # Unbatched
            x_seq = x_seq.unsqueeze(0)

        B, T, N, F = x_seq.size()

        # Reshape for transformer: [B*N, T, F]
        x_reshaped = x_seq.permute(0, 2, 1, 3).reshape(B * N, T, F)
        x_proj = self.transformer_input_proj(x_reshaped)
        x_encoded = self.transformer_encoder(x_proj)
        x_decoded = self.proj_back(x_encoded)

        # Reshape back to [B, T, N, F]
        x_seq_transformed = x_decoded.reshape(B, N, T, F).permute(0, 2, 1, 3)

        # Feed through GConvGRU
        h = torch.zeros(B, N, self.gconv_gru.hidden_channels, device=x_seq.device)
        for t in range(T):
            h = self.gconv_gru(x_seq_transformed[:, t], h, edge_index, edge_weight)

        out = self.linear(self.relu(h))  # [B, N, out_channels]
        return out
