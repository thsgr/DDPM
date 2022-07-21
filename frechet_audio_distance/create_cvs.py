import os
import numpy as np
from glob import glob
import random
from pathlib import Path




def create_cvs_file(files_dir):

    name = files_dir.rsplit("/")[1]
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filenames = glob(f'{files_dir}/**/*.wav', recursive=True)
    filenames = np.array(filenames)

    file = f"{dir_path}/audio_cvs/{name}.cvs"
    np.savetxt(file,
                filenames,
                delimiter=", ",
                fmt='% s')
    print(f'File {name}.cvs have been created \n')

    # Make sure stats folder exists
    if not Path(f"{dir_path}/stats").exists():
        os.mkdir(f"{dir_path}/stats")
        print(f"created Stats folder under {dir_path} \n")
    
    stat_path = f"{dir_path}/stats/{name}_stats"
    return file, stat_path

def create_manually_cvs_file(files_dir, name):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filenames = glob(f'{files_dir}/**/*.wav', recursive=True)
    filenames = np.array(filenames)

    file = f"{dir_path}/audio_cvs/{name}.cvs"
    np.savetxt(file,
                filenames,
                delimiter=", ",
                fmt='% s')
    print(f'File {name}.cvs have been created \n')

    # Make sure stats folder exists
    if not Path(f"{dir_path}/stats").exists():
        os.mkdir(f"{dir_path}/stats")
        print(f"created Stats folder under {dir_path} \n")



if __name__=="__main__":
    create_manually_cvs_file('/home/thomas/datasets/drums_preprocessed/train', 'drums')
