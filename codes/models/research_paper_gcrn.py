import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU


class GCRN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K=2, normalization="sym", out=1, num_classes=2,
                 task_type='regression'):
        super(GCRN, self).__init__()

        # Camada GConvGRU
        self.recurrent = GConvGRU(
            in_channels=in_channels,
            out_channels=out_channels,
            K=K,
            normalization=normalization,
            bias=True
        )
        self.recurrent.node_dim = 1

        # Atributos de configuração
        self.task_type = task_type
        self.out = out
        self.num_classes = num_classes

        # Camada linear
        if task_type == 'classification':
            # Camada linear para classificação multiclasse
            self.linear = torch.nn.Linear(in_features=out_channels, out_features=num_classes * out)
        else:
            self.linear = torch.nn.Linear(in_features=out_channels, out_features=out)

    def forward(self, x, edge_index, edge_weight, explainer=False):
        # x: [T, N, F]
        T, N, F = x.size()
        h = self.recurrent(x, edge_index, edge_weight)  # [N, H]
        h = F.relu(h)
        h = self.linear(h)  # [N, out]
        return h  # [N, out]


