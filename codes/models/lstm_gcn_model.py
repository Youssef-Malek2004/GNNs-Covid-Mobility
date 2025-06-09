import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.conv_i = GCNConv(in_channels + hidden_channels, hidden_channels)
        self.conv_f = GCNConv(in_channels + hidden_channels, hidden_channels)
        self.conv_o = GCNConv(in_channels + hidden_channels, hidden_channels)
        self.conv_g = GCNConv(in_channels + hidden_channels, hidden_channels)

    def forward(self, x, h, c, edge_index, edge_weight=None):
        B, N, _ = x.size()
        h_next, c_next = [], []

        for b in range(B):
            xh = torch.cat([x[b], h[b]], dim=-1)  # [N, F+H]
            i = torch.sigmoid(self.conv_i(xh, edge_index, edge_weight))
            f = torch.sigmoid(self.conv_f(xh, edge_index, edge_weight))
            o = torch.sigmoid(self.conv_o(xh, edge_index, edge_weight))
            g = torch.tanh(self.conv_g(xh, edge_index, edge_weight))

            c_new = f * c[b] + i * g
            h_new = o * torch.tanh(c_new)

            c_next.append(c_new)
            h_next.append(h_new)

        h_stack = torch.stack(h_next, dim=0)  # [B, N, H]
        c_stack = torch.stack(c_next, dim=0)  # [B, N, H]
        return h_stack, c_stack


class GCRN_LSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCRN_LSTM, self).__init__()
        self.gconv_lstm = GConvLSTMCell(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_seq, edge_index, edge_weight=None):
        """
        x_seq: [B, T, N, F]
        Returns: [B, N, out_channels]
        """
        if x_seq.dim() == 3:  # Unbatched: [T, N, F]
            x_seq = x_seq.unsqueeze(0)

        B, T, N, F = x_seq.size()
        h = torch.zeros(B, N, self.gconv_lstm.hidden_channels, device=x_seq.device)
        c = torch.zeros(B, N, self.gconv_lstm.hidden_channels, device=x_seq.device)

        for t in range(T):
            h, c = self.gconv_lstm(x_seq[:, t], h, c, edge_index, edge_weight)

        out = self.linear(self.relu(h))  # [B, N, out_channels]
        return out
