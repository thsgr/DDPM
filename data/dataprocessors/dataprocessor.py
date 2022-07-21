import torch
from torch import nn
from torch.functional import Tensor
from data.dataloaders.dataloader_generator import DataloaderGenerator
from diffusion.utils import cuda_variable
import numpy as np

class DataProcessor(nn.Module):
    """
    Abstract class used for preprocessing and postprocessing
    Can contain trainable parameters
    
    Preprocessing: from ? -> (batch_size, num_events)
    Postprocessing: from (batch_size, num_events) -> ?      
    
    ? designates what comes out of the dataloaders
    """
    def __init__(self):
        super().__init__()

    def preprocess(self, data:torch.Tensor)->torch.Tensor:
        return data

    def postprocess(self, data:torch.Tensor)->torch.Tensor:
        return data


class MuLawDataProcessor(nn.Module):
    """
    Class used for preprocessing and postprocessing
    Can contain trainable parameters
    
    Preprocessing: from ? -> (batch_size, num_events)
    Postprocessing: from (batch_size, num_events) -> ?      
    
    ? designates what comes out of the dataloaders
    """
    def __init__(self):
        super().__init__()
        self.mu = 255

    def preprocess(self, data:torch.Tensor)->torch.Tensor:
        self.scaler = max(data.max(), -data.min())
        if self.scaler > 0:
            data = data / self.scaler
        data = torch.sign(data)*torch.log(1 + self.mu * torch.abs(data)) / np.log(self.mu + 1)
        return data

    def postprocess(self, data:torch.Tensor)->torch.Tensor:
        data = torch.sign(data)*(torch.pow(1 + self.mu, torch.abs(data)) - 1) / self.mu
        return data
