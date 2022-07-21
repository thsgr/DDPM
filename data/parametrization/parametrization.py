from ast import Not
import torch
from torch import nn
from torch.functional import Tensor
import numpy as np


class DataParametrization(nn.Module):

    def __init__(self, objective):
        super().__init__()
        self.objective = objective

    def parametrize(self, v_t, l_t, sigma_vv_t, alpha_prime):
        raise NotImplementedError

    def coeff_objective(self, l_t, beta, Gamma_inv):
        if self.objective == "MLE":
            lambda_t = beta / Gamma_inv
            return lambda_t * l_t ** 2
        else:
            return 1
