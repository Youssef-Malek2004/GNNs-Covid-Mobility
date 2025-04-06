import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GConvGRU
import torch.nn.functional as F


class GCRN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, K=2, normalization="sym", task_type="regression"):
        super(GCRN, self).__init__()

        self.hidden_channels = hidden_channels
        self.task_type = task_type

        # Temporal GConvGRU from PyTorch Geometric Temporal
        self.recurrent = GConvGRU(in_channels=in_channels, out_channels=hidden_channels, K=K, normalization=normalization)

        # Final linear layer after GRU
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_seq, edge_index, edge_weight=None):
        """
        x_seq: [B, T, N, F] or [T, N, F]
        Returns: [B, N, out_channels]
        """
        if x_seq.dim() == 3:
            x_seq = x_seq.unsqueeze(0)

        B, T, N, feat_dim = x_seq.size()
        output = []

        for b in range(B):
            h = None
            for t in range(T):
                x_t = x_seq[b, t]
                h = self.recurrent(x_t, edge_index, edge_weight, h)
            out = self.linear(F.relu(h))  # [N, out_channels]
            output.append(out)

        return torch.stack(output, dim=0)  # [B, N, out_channels]
