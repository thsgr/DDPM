from pathlib import Path
from contextlib import nullcontext

from torch import autograd
from data.dataloaders import DataloaderGenerator
from data.dataprocessors import DataProcessor
from handlers.ema import ExponentialMovingAverage
from data.parametrization import DataParametrization
from diffusion.samplers.sampler import Sampler
from diffusion.utils import is_main_process, rms
from diffusion.sdes.sde import SDE

import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from tqdm import tqdm
from itertools import islice

class Handler:
    """
    General class to handle the training of a DistributedDataParallel model
    where model is a DiffusionModel
    Inherit from this class to implement a specific epoch
    """

    def __init__(self, model: DistributedDataParallel,
                 dataloader_generator: DataloaderGenerator,
                 data_processor: DataProcessor,
                 sde: SDE,
                 sampler: Sampler,
                 ema_rate: float,
                 parametrization: DataParametrization,
                 ) -> None:

        self.sde = sde
        self.sampler = sampler
        self.model = model
        self.dataloader_generator = dataloader_generator
        self.data_processor = data_processor
        self.ema = ExponentialMovingAverage(
            self.parameters(), decay=ema_rate)
        self.parametrization = parametrization
        self.state = dict(model=self.model, ema=self.ema, step=0)

        # optim
        self.optimizer = None
        self.scheduler = None
        self.autocast = None
        self.scaler = None

    def init_optimizers(self, lr=1e-3, train_16_bits=False):
        self.optimizer = torch.optim.Adam(list(self.parameters()), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=150000, gamma=0.5
        )

    # ==== Wrappers ====

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()


    # ==== Save and Load methods ====
    def __repr__(self):
        return self.model.module.__repr__()

    def save(self, save_fp):
        if hasattr(self, 'load_ema'):
            self.ema.restore(self.model.parameters())

        saved_state = {
            'model': self.state['model'].state_dict(),
            'ema': self.state['ema'].state_dict(),
            'step': self.state['step']
        }
        save_fp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved_state, save_fp)

    def prune_checkpoints(self, model_dir, num_models_to_save=None):
        if num_models_to_save is not None:
            checkpoints_fp = self.get_checkpoints_newest_first(model_dir)
            for fp in checkpoints_fp[num_models_to_save:]:
                fp.unlink()

    def get_checkpoints_newest_first(self, model_dir):
        return sorted(Path(model_dir).glob('weights_*.pt'),
                      key=lambda fp: -fp.stat().st_ctime)

    def load(self, model_dir, checkpoint_id=None, load_ema=False):
        
        if torch.cuda.is_available():
            map_location = {'cuda:0': f'cuda:{dist.get_rank()}'}
        else:
            map_location = {'cpu': 'cpu'}
        
        print(f'Loading models {self.__repr__()} \n')

        self.load_ema = load_ema
        if checkpoint_id is None:
            checkpoints = self.get_checkpoints_newest_first(model_dir)
            if checkpoints:
                checkpoint_fp = checkpoints[0]
            else:
                return False
        else:
            checkpoint_fp = model_dir / f'weights_{checkpoint_id}.pt'

        try:
            loaded_state = torch.load(checkpoint_fp, map_location=map_location)
            
            if load_ema:
                self.state['ema'].load_state_dict(loaded_state['ema'])
                self.state['model'].load_state_dict(loaded_state['model'])
                self.ema.store(self.model.parameters())
                self.ema.copy_to(self.model.parameters())
                print('EMA parameters loaded in model \n')

            else:
                self.state['model'].load_state_dict(loaded_state['model'])
                self.state['ema'].load_state_dict(loaded_state['ema'])

            self.state['step'] = loaded_state['step']
            print(f'Model {checkpoint_fp} loaded \n')
            return True
        except FileNotFoundError:
            return False


    def train_model(self,
                    batch_size,
                    num_batches=None,
                    num_epochs=10,
                    lr=1e-3,
                    num_workers=0,
                    n_bins=10,
                    num_epochs_between_savings=20,
                    train_16_bits=False,
                    num_models_to_save=None,
                    run_logger=None,
                    **kwargs
                    ):

        # best_val = 1e8
        self.init_optimizers(lr=lr, train_16_bits=train_16_bits)

        # context makes sure logger propery closes on exit
        if run_logger:
            logger_context = run_logger
        else:
            logger_context = nullcontext()

        with logger_context:
            for epoch_id in range(num_epochs):
                (generator_train, generator_val, _) = self.dataloader_generator.dataloaders(
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle_val=True)

                monitored_quantities_train = self.epoch(
                    data_loader=generator_train,
                    train=True,
                    num_batches=num_batches,
                    n_bins=n_bins
                )
                del generator_train

                if epoch_id % 10 == 0:
                    with torch.no_grad():
                        monitored_quantities_val = self.epoch(
                            data_loader=generator_val,
                            train=False,
                            num_batches=num_batches // 2 if num_batches is not None else None,
                            n_bins=n_bins
                        )
                        del generator_val
                else:
                    monitored_quantities_val = None

                if run_logger:
                    run_logger.log_epoch(epoch_id, monitored_quantities_train,
                                         monitored_quantities_val)

                if epoch_id % num_epochs_between_savings == 0:
                    if run_logger:
                        save_fp = run_logger.model_dir / f'weights_{epoch_id}.pt'
                        self.save(save_fp)
                        self.prune_checkpoints(run_logger.model_dir, num_models_to_save)
                    dist.barrier()


    def epoch(
        self,
        data_loader,
        train=True,
        num_batches=None,
        n_bins=10,
    ):
        num_elems_in_bins = np.zeros(n_bins)
        sum_loss_in_bins = np.zeros(n_bins)
        cum_grad_norm = 0

        if train:
            self.train()
        else:
            self.eval()

        iterator = enumerate(islice(data_loader, num_batches))
        if is_main_process():
            iterator = tqdm(iterator, ncols=80)

        for sample_id, tensor_dict in iterator:

            self.optimizer.zero_grad()

            # with autograd.detect_anomaly():
            #     forward_pass = self.forward_pass(tensor_dict,
            #                                         n_bins)

            forward_pass = self.forward_pass(tensor_dict, n_bins)

            loss = forward_pass['loss']
            if train:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.parameters(), 5.0)
                self.optimizer.step()
                self.scheduler.step()
                self.state['step'] += 1
                self.state['ema'].update(self.parameters())

            # Monitor quantities
            monitored_quantities = forward_pass['monitored_quantities']

            num_elems_in_bins += monitored_quantities['num_elems_in_bins']
            sum_loss_in_bins += monitored_quantities['sum_loss_in_bins']
            if train:
                cum_grad_norm += grad_norm

            del loss
        # renormalize monitored quantities
        loss_in_bins = np.divide(
            sum_loss_in_bins, num_elems_in_bins)
        if train:
            mean_grad_norm = cum_grad_norm / num_batches

        dic_loss_in_bins = dict((f"loss_in_bin_{bin_id}", bin_loss)
                                for bin_id, bin_loss in enumerate(loss_in_bins))

        if train:
            return (dic_loss_in_bins, mean_grad_norm)
        else:
            return dic_loss_in_bins

    def generate(self, cond=None, num_samples=None, batch_size=None, guidance=1.0):
        self.model.eval()

        sample_kwargs = {}
        if cond is None:
            if num_samples:
                sample_kwargs['num_samples'] = num_samples
            if batch_size:
                sample_kwargs['batch_size'] = batch_size
        else:
            prep_cond = self.data_processor.preprocess(cond)
            if batch_size:
                prep_cond = torch.tile(prep_cond, (batch_size, 1))
                cond = torch.tile(cond, (batch_size,))
            cond = cond.reshape((1, -1))
            sample_kwargs['cond'] = prep_cond
            sample_kwargs['guidance'] = guidance

        x = self.sampler.sample(**sample_kwargs)
        x = self.data_processor.postprocess(x).cpu()
        return x


    def generate_save(self, config):
        
        if config.input_audio:
            import librosa
            in_audio, _ = librosa.load(config.input_audio, sr=config.fs)
            audio_len = in_audio.shape[-1]
            # ensure 2d
            in_audio = in_audio.reshape((-1, audio_len))
            # truncate to desired length
            in_audio = in_audio[:, :config.num_samples]
            # mono
            cond = torch.from_numpy(in_audio).mean(dim=0)
        else:
            cond = None

        out_audio = self.generate(cond, config.num_samples, config.batch_size, config.guidance)
        if cond is None:
            out = out_audio
        else:
            n_channels = len(in_audio)
            out_audio = torch.tile(out_audio, (n_channels, 1))
            in_audio = torch.from_numpy(in_audio)
            in_audio = torch.tile(in_audio, (1, config.batch_size))

            # make RMS of output equal to RMS of input
            rms_in = rms(in_audio)
            rms_out = rms(out_audio)
            if rms_in > 0:
                out_audio *= (rms_in/rms_out)

            pan = 0
            left = (1 - pan) * in_audio + pan * out_audio
            right = pan * in_audio + (1 - pan)* out_audio

            out = torch.cat((left, right))
            v_max = out.abs().max()
            if v_max > 1:
                out /= v_max

        if config.output_audio:
            import torchaudio
            torchaudio.save(config.output_audio, out, config.fs, channels_first=True)

    def forward_pass(self, tensor_dict, n_bins):
        raise NotImplementedError
