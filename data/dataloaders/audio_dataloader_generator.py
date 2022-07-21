import os
import torch

from data.dataloaders.dataloader_generator import DataloaderGenerator
from data.datasets.audio_dataset import AudioDataset


class AudioDataloaderGenerator(DataloaderGenerator):
    def __init__(self, dataloader_generator_kwargs):
        self.train_paths = dataloader_generator_kwargs['train_paths']
        self.val_paths = dataloader_generator_kwargs['val_paths']
        self.test_paths = dataloader_generator_kwargs['test_paths']
        self.audio_length = dataloader_generator_kwargs['audio_length']

    def dataloaders(self, batch_size, num_workers=0, shuffle_train=True,
                    shuffle_val=False):

        train_dataset = AudioDataset(self.train_paths, self.audio_length)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        dataloader_train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=None,
            shuffle=False,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True)

        val_dataset = AudioDataset(self.val_paths, self.audio_length)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset)
        dataloader_val = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=None,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True)

        test_dataset = AudioDataset(self.test_paths, self.audio_length)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset)
        dataloader_test = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=None,
            shuffle=False,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False)

        return (dataloader_train, dataloader_val, dataloader_test)
