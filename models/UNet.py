import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import GammaBeta, RFFBlock, DBlock, UBlock, Conv1d


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = Conv1d(2, 32, 5, padding=2)
        self.embedding = RFFBlock(
            input_dim=1, rff_dim=32, mlp_hidden_dims=[128, 256, 512])
        self.downsample = nn.ModuleList([
            DBlock(32, 128, 2),
            DBlock(128, 128, 2),
            DBlock(128, 256, 3),
            DBlock(256, 512, 5),
            DBlock(512, 512, 5),
        ])
        self.gamma_beta = nn.ModuleList([
            GammaBeta(input_dim=512, output_dim=128),
            GammaBeta(input_dim=512, output_dim=128),
            GammaBeta(input_dim=512, output_dim=256),
            GammaBeta(input_dim=512, output_dim=512),
            GammaBeta(input_dim=512, output_dim=512),
        ])
        self.upsample = nn.ModuleList([
            UBlock(512, 512, 5, [1, 2, 4, 8]),
            UBlock(512, 256, 5, [1, 2, 4, 8]),
            UBlock(256, 128, 3, [1, 2, 4, 8]),
            UBlock(128, 128, 2, [1, 2, 4, 8]),
            UBlock(128, 128, 2, [1, 2, 4, 8]),
        ])
        self.last_conv = Conv1d(128, 1, 3, padding=1)

    def forward(self, x, noise_scale):
        """
        x is (batch_size, length, 2) contains position and velocity
        """
        # channel first 
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        downsampled = []
        noise_scale = self.embedding(noise_scale)

        for film, layer in zip(self.gamma_beta, self.downsample):
            gamma, beta = film(noise_scale)
            x = layer(x, gamma, beta)
            downsampled.append(x)

        for layer, x_dblock in zip(self.upsample, reversed(downsampled)):
            x = layer(x, x_dblock)

        # TODO add attention
        # TODO no leaky relu
        x = self.last_conv(x)
        x = x.squeeze(1)
        return x
