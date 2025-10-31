import torch
import torch.nn as nn
import torch.nn.functional as F

class SeriesStationarization:
    def normalize(self, x):  # [B, T, N, F]
        mu = x.mean(dim=1, keepdim=True)               # [B, 1, N, F]
        std = x.std(dim=1, keepdim=True) + 1e-6        # [B, 1, N, F]
        return (x - mu) / std, mu, std

    def denormalize(self, y, mu, std):  # y: [B, N]
        return y * std.squeeze(1).squeeze(-1) + mu.squeeze(1).squeeze(-1)

class DeStationaryAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = d_model ** -0.5
        self.tau = nn.Linear(1, 1)

    def forward(self, Q, K, V, std):  # [B, T, N, D]
        B, T, N, D = Q.shape

        # Center Q, K over time
        Q = Q - Q.mean(dim=1, keepdim=True)
        K = K - K.mean(dim=1, keepdim=True)

        # Tau: [B, 1, N, 1]
        tau = torch.sigmoid(self.tau(std.mean(dim=1, keepdim=True))) * 2

        # Reshape for attention per node
        Q = Q.permute(0, 2, 1, 3)  # [B, N, T, D]
        K = K.permute(0, 2, 1, 3)  # [B, N, T, D]
        V = V.permute(0, 2, 1, 3)  # [B, N, T, D]

        # Compute attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, N, T, T]
        attn = attn * tau.squeeze(-1).permute(0, 2, 1)[:, :, None]  # broadcast tau: [B, N, 1]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)  # [B, N, T, D]
        out = out.permute(0, 2, 1, 3)  # [B, T, N, D]
        return out


class NonStationaryBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = DeStationaryAttention(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, std):  # x: [B, T, N, D]
        h = self.attn(x, x, x, std)                       # [B, T, N, D]
        x = self.norm1(x + h)
        x = self.norm2(x + self.ff(x))
        return x

class MinimalSpatioTemporalModel(nn.Module):
    def __init__(self, input_dim=1, d_model=32, num_layers=2):
        super().__init__()
        self.station = SeriesStationarization()
        self.embed = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            NonStationaryBlock(d_model) for _ in range(num_layers)
        ])
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):  # x: [B, T, N, F]
        B, T, N, F = x.shape

        x_norm, mu, std = self.station.normalize(x)      # [B, T, N, F]
        h = self.embed(x_norm)                           # [B, T, N, D]

        for block in self.blocks:
            h = block(h, std)

        out = self.proj(h[:, -1])                        # [B, N, 1]
        out = out.squeeze(-1)                            # [B, N]
        out = self.station.denormalize(out, mu, std)     # [B, N]

        return out
