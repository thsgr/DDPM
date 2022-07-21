import torch
from torch.functional import Tensor
from diffusion.utils import cuda_variable
import numpy as np
import torch.distributed as dist


class SDE(torch.nn.Module):
    def __init__(self, mle_training):
        self.mle_training = mle_training
        super().__init__()

    def l(self, t: torch.Tensor):
        """[summary]

        Args:
            t ([type]): [description]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def sigma(self, t: torch.Tensor):
        raise NotImplementedError

    def mu(self, u_t: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError

    def perturb(self, x, t, noise_x=None, noise_v=None):
        raise NotImplementedError

    def likelihood_weighter(self, t: torch.Tensor):
        raise NotImplementedError

    def sigma(self, t: torch.Tensor):
        raise NotImplementedError

    def mean(self, t: torch.Tensor):
        raise NotImplementedError
    
    def perturb(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        mean = self.mean(t)
        sigma = self.sigma(t)
        return mean * x + sigma * noise


class DDPMSde(SDE):
    def __init__(self):
        self.t_min = 0.0
        self.t_max = 1.0

    def sigma(self, t: torch.Tensor):
        return torch.sigmoid(22.31 * t - 18.42)**0.5

    def mean(self, t: torch.Tensor):
        return (1 - self.sigma(t)**2)**0.5

    def beta(self, t: torch.Tensor):
        return 22.31 * torch.sigmoid(22.31 * t - 18.42)

    def g(self, t: torch.Tensor):
        return self.beta(t)**0.5

