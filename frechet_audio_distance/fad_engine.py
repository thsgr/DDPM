from glob import glob
import torchaudio
import torch
from pathlib import Path
import os
from .create_cvs import create_cvs_file
from .fad_utils import save_split
import subprocess

class FAD():

    def __init__(self, gen_config, fad_config):
        self.gen_config = gen_config
        self.fad_config = fad_config

    def pipeline_cvs(self):
        
        # Splitting audio in *batch_size* different files
        audio, _ = torchaudio.load(self.gen_config.output_audio)
        save_split(self.gen_config, audio)
        
        # creating cvs file
        input_files_path = self.gen_config.output_audio.rsplit(".")[0]
        self.input_files, self.stats = create_cvs_file(input_files_path)

    def compute_fad(self):

        # if cvs file doesn't exist, create it using gen_config output
            # it creates a subfolder with splitted files
            if self.fad_config.input_files is None:
                self.pipeline_cvs()
            
            if not Path(self.fad_config.background_stats).exists():
                raise FileExistsError(
                    f'{self.fad_config.background_stats} does not exist : \n'
                    "Please compute stats on dataset separately and store it in stats folder"
                    )

            # Create stats
            os.system("python -m frechet_audio_distance.create_embeddings_main" +
                      f" --input_files {self.input_files}" + 
                      f" --stats {self.stats}")
            
            # Compute FAD
            os.system("python -m frechet_audio_distance.compute_fad" +
                      f" --background_stats {self.fad_config.background_stats}" + 
                      f" --test_stats {self.stats}")
