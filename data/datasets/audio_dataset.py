from glob import glob
import numpy as np
import os
import random
import torch
import torchaudio


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, paths, audio_length):
        super().__init__()
        self.audio_length = audio_length
        self.filenames = []
        for path in paths:
            self.filenames += glob(f'{path}/**/*.wav', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        signal, _ = torchaudio.load(audio_filename)

        return signal[0, :self.audio_length]
