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
            xh = torch.cat([x[b], h[b]], dim=-1)  # [N, F+H]
            z = torch.sigmoid(self.conv_z(xh, edge_index, edge_weight))
            r = torch.sigmoid(self.conv_r(xh, edge_index, edge_weight))
            rh = torch.cat([x[b], r * h[b]], dim=-1)
            h_tilde = torch.tanh(self.conv_h(rh, edge_index, edge_weight))
            h_new = (1 - z) * h[b] + z * h_tilde
            h_next.append(h_new)

        return torch.stack(h_next, dim=0)  # [B, N, H]



class GCRN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCRN, self).__init__()
        self.gconv_gru = GConvGRUCell(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_seq, edge_index, edge_weight=None):
        """
        x_seq: [B, T, N, F] – batched input
        Returns: [B, N, out_channels]
        """
        if x_seq.dim() == 3:  # Unbatched: [T, N, F]
            x_seq = x_seq.unsqueeze(0)  # -> [1, T, N, F]

        B, T, N, F = x_seq.size()
        h = torch.zeros(B, N, self.gconv_gru.hidden_channels, device=x_seq.device)

        for t in range(T):
            h = self.gconv_gru(x_seq[:, t], h, edge_index, edge_weight)  # [B, N, H]

        out = self.linear(self.relu(h))  # [B, N, out_channels]
        return out

