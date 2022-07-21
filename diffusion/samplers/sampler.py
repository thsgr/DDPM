import torch
from torch.nn.parallel.distributed import DistributedDataParallel
from diffusion.sdes.sde import SDE
from data.parametrization.parametrization import DataParametrization
from diffusion.utils import cuda_variable
from tqdm import tqdm
import numpy as np
import torch.distributed as dist

class Sampler():

    def __init__(
        self, 
        num_steps, 
        sde: SDE, 
        model: DistributedDataParallel, 
        parametrization: DataParametrization,
        ) -> None:

        self.sde = sde
        self.parametrization = parametrization
        self.model = model
        self.num_steps = num_steps

    def _to_numpy(self, tens):
            a = np.array(tens.cpu().numpy(), dtype=np.float64)
            return a 

    def to_tensor(self, array):
        return torch.tensor(array, dtype=torch.float64).to(device=f'cuda:{dist.get_rank()}')

    def sample(self):
        raise NotImplementedError

    def striding(self):
        raise NotImplementedError



class EMSampler(Sampler):
    """
    DDPM-like discretization of the SDE as in https://arxiv.org/abs/2107.00630
    This is the most precise discretization
    """

    def __init__(self, num_steps, sde, model, parametrization):

        super().__init__(num_steps, sde, model, parametrization)

    def striding(self):

        nb_steps = self.num_steps
        t_schedule = torch.arange(0, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        sigma_schedule = self.sde.sigma(t_schedule)
        m_schedule = self.sde.mean(t_schedule)

        return sigma_schedule, m_schedule

    def sample(self, num_samples = 21000, batch_size=1):
        
        audio = cuda_variable(torch.randn(batch_size, num_samples))
        nb_steps = self.num_steps

        with torch.no_grad():

            sigma, m = self.striding()

            for n in tqdm(range(nb_steps - 1, 0, -1)):
                # begins at t = 1 (n = nb_steps - 1)
                # stops at t = 2/nb_steps (n=1)

                audio = m[n-1] / m[n] * audio + (m[n] / m[n-1] * (sigma[n-1])**2 / sigma[n] - m[n-1] / m[n] * sigma[n]) * \
                    self.model(audio, sigma[n])

                if n > 0:  # everytime
                    noise = torch.randn_like(audio)
                    audio += sigma[n-1]*(1 - (sigma[n-1]*m[n] /
                                              (sigma[n]*m[n-1]))**2)**0.5 * noise

            # The noise level is now sigma(1/nb_steps) = sigma[0]
            # Jump step
            audio = (audio - sigma[0] * self.model(audio,
                                                   sigma[0])) / m[0]
        
        audio = audio.reshape(1, batch_size * audio.size(1))
        audio = audio.to(dtype=torch.float32)
        
        return audio

