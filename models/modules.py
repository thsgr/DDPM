import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RFFBlock(nn.Module):
    def __init__(self, input_dim=1, rff_dim=32, mlp_hidden_dims: list = None):
        """[summary]
        Args:
            input_dim (int, optional): [description]. Defaults to 1.
            rff_dim (int, optional): [needs to be half of desired output shape]. Defaults to 32.
            mlp_hidden_dims (list, optional): [dimension of MLP if need to use one after RFF]. Defaults to None.
        """
        super().__init__()
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([input_dim, rff_dim]), requires_grad=False)
        if mlp_hidden_dims:
            assert len(mlp_hidden_dims) > 0
            mlp_hidden_dims.insert(0, 2*rff_dim)
            self.MLP = MLP(mlp_dimensions=mlp_hidden_dims, non_linearity=nn.ReLU(), last_non_linearity=nn.ReLU())
        else:
            self.MLP = None

    def forward(self, std_step):
        """
        Arguments:
          std_step:
              (shape: [B, input_shape], dtype: float32)
        Returns:
          x: embedding of sigma
              (shape: [B, 2 * rff_dim], dtype: float32)
        """
        x = self._build_RFF_embedding(std_step)
        if self.MLP:
            x = self.MLP(x)
        return x

    def _build_RFF_embedding(self, std_step):
        """
        Arguments:
          std_step:
              (shape: [..., 1], dtype: float32)
        Returns:
          table:
              (shape: [..., 64], dtype: float32)
        """
        freqs = self.RFF_freq
        table = 2 * np.pi * torch.einsum('...i,ij->...j', std_step, freqs)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=-1)
        return table


class MLP(nn.Module):
    def __init__(self, mlp_dimensions: list, non_linearity=nn.LeakyReLU(), last_non_linearity=None):
        super().__init__()
        assert len(mlp_dimensions) > 0
        self.layers = nn.ModuleList([
            nn.Linear(mlp_dimensions[i-1], mlp_dimensions[i]) for i in range(1, len(mlp_dimensions))
        ])
        self.non_linearity = non_linearity
        self.last_non_linearity = last_non_linearity
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.non_linearity(l(x))
        y = self.layers[-1](x)
        if self.last_non_linearity:
            y = self.non_linearity(y)
        return y


class GammaBeta(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_layer = nn.Linear(input_dim, 2 * output_dim)

    def forward(self, input):
        input = self.output_layer(input)
        input = input.unsqueeze(-1)
        gamma, beta = torch.chunk(input, 2, dim=1)
        return gamma, beta

class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)


class UBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor, dilation):
        super().__init__()
        assert isinstance(dilation, (list, tuple))
        assert len(dilation) == 4

        self.factor = factor
        self.residual_dense = Conv1d(input_size, hidden_size, 1)
        self.convs = nn.ModuleList([
            Conv1d(2 * input_size, hidden_size, 3,
                   dilation=dilation[0], padding=dilation[0]),
            Conv1d(hidden_size, hidden_size, 3,
                   dilation=dilation[1], padding=dilation[1]),
            Conv1d(hidden_size, hidden_size, 3,
                   dilation=dilation[2], padding=dilation[2]),
            Conv1d(hidden_size, hidden_size, 3,
                   dilation=dilation[3], padding=dilation[3])
        ])

    def forward(self, x, x_dblock):
        size = x.shape[-1] * self.factor

        residual = F.interpolate(x, size=size)
        residual = self.residual_dense(residual)

        x = torch.cat([x, x_dblock], dim=1)
        x = F.leaky_relu(x, 0.2)
        x = F.interpolate(x, size=size)
        for layer in self.convs:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)
        return x + residual


class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor):
        super().__init__()
        self.factor = factor
        self.residual_dense = Conv1d(input_size, hidden_size, 1)
        self.layer_1 = Conv1d(input_size, hidden_size,
                              3, dilation=1, padding=1)
        self.convs = nn.ModuleList([
            Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2),
            Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4),
            Conv1d(hidden_size, hidden_size, 3, dilation=8, padding=8),

        ])

    def forward(self, x, gamma, beta):
        size = x.shape[-1] // self.factor

        residual = self.residual_dense(x)
        residual = F.interpolate(residual, size=size)

        x = F.interpolate(x, size=size)
        x = F.leaky_relu(x, 0.2)
        x = self.layer_1(x)
        x = gamma * x + beta
        for layer in self.convs:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)

        return x + residual
