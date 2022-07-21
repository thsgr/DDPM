from torch.nn.parallel.distributed import DistributedDataParallel
from data.dataloaders import DataloaderGenerator
from data.dataprocessors.dataprocessor import DataProcessor
from handlers.handler import Handler
from data.parametrization.parametrization import DataParametrization
from diffusion.sdes.sde import SDE
from diffusion.samplers.sampler import Sampler
import torch
import torch.nn.functional as F
from diffusion.utils import cuda_variable
import numpy as np


class DDPMHandler(Handler):
    def __init__(
        self,
        model: DistributedDataParallel,
        dataloader_generator: DataloaderGenerator,
        data_processor: DataProcessor,
        sde: SDE,
        sampler: Sampler,
        ema_rate: float,
        parametrization: DataParametrization,

    ) -> None:

        super().__init__(
            model=model,
            dataloader_generator=dataloader_generator,
            data_processor=data_processor,
            sde=sde,
            sampler=sampler,
            ema_rate=ema_rate,
            parametrization=parametrization,
        )

    def forward_pass(self, tensor_dict, n_bins):

        if isinstance(tensor_dict, dict):
            x = tensor_dict['x']
        else:
            x = tensor_dict

        x = self.data_processor.preprocess(x)


        batch_size, length = x.size()
        # Limit of time under which no sampling
        lim = 1e-5

        # Make sure the batch covers the range [0, 1] uniformly with a random offset
        t = torch.rand(batch_size, 1, device=x.device)
        t = (self.sde.t_max - self.sde.t_min) * t + self.sde.t_min
        
        noise = torch.randn_like(x)

        noisy_audio = self.sde.perturb(x, t, noise)
        sigma = self.sde.sigma(t)
        predicted = self.model(noisy_audio, sigma)

        noise = cuda_variable(noise)

        vectorial_loss =  F.mse_loss(predicted, noise, reduction='none')
        loss = vectorial_loss.mean()
        vectorial_loss = vectorial_loss.detach()

        monitored_quantities = self.compute_monitored_quantities(
            vectorial_loss, t, n_bins
        )

        return dict(
            loss=loss,
            monitored_quantities=monitored_quantities,
        )

    def compute_monitored_quantities(self, vectorial_loss, t, n_bins):
        """
        n_bins : integer
        """

        # TODO must only contain scalars
        # create entries like {'sum': , 'count': } so that averaging later can be done correctly
        if vectorial_loss.size() != t.size():
            # if there is one t per sample
            vectorial_loss = torch.mean(vectorial_loss, 1)

        # TODO : check if the commented lines are necessary

        vectorial_loss = vectorial_loss.cpu().numpy()
        vectorial_loss = np.reshape(vectorial_loss, -1)

        t = t.cpu().numpy()
        t = np.reshape(t, -1)

        num_elems_in_bins = np.zeros(n_bins)
        sum_loss_in_bins = np.zeros(n_bins)

        bins_index = np.trunc(n_bins * t)
        bins_index = bins_index.astype(int)

        for bin_id in range(n_bins):
            num_elems_in_bins[bin_id] = (bins_index == bin_id).sum()
            sum_loss_in_bins[bin_id] = vectorial_loss[bins_index == bin_id].sum()

        monitored_quantities = dict(
            num_elems_in_bins=num_elems_in_bins, sum_loss_in_bins=sum_loss_in_bins
        )
        return monitored_quantities
