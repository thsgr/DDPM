"""
@author: Thomas Segré
"""
import os
import sys
from pathlib import Path
import warnings
import click
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from getters import generate_configs, get_logger, load_handler
from diffusion.utils import init_group_process, AttrDict, load_yaml_config
from frechet_audio_distance.getters import get_fad_engine


@click.command()
@click.option('-t', '--train', is_flag=True)
@click.option('-l', '--load', is_flag=True)
@click.option('-c', '--config', default="configs/default_config.yaml", type=click.Path(exists=True))
@click.option('-n', '--num_steps', type=int, help="Number of generating steps")
@click.option('-w', '--num_workers', type=int)
@click.option('-d', '--dataprocessor_type', type=str)
@click.option('-s', '--sde', type=str, help="SDE name")
@click.option('-h', '--sampler_name', type=str, help="Sampling chosen")
@click.option('-g', '--generate_num_samples', type=int)
@click.option('-k', '--striding', type=str, help="Type of striding in generation")
@click.option('-i', '--input_audio', type=click.Path(exists=True), help="Path to input audio")
@click.option('-o', '--output_audio', type=click.Path(exists=False), help="Path to output audio (file should not exist)")
@click.option('-b', '--generate_batch_size', type=int, help="Number of audio segments to generate")
@click.option('-f', '--fs', type=int)
@click.option('-q', '--guidance', type=float)
@click.option('-m', '--diffusion_model_type', type=str, help="Neural Network Model")
@click.option('-r', '--lr', type=float, help="Learning Rate")
@click.option('-u', '--batch_size', type=int)
@click.option('-v', '--num_batches', type=int)
@click.option('-x', '--ema_rate', type=float)
@click.option('-y', '--load_ema', type=bool, is_flag=True, default=True)
@click.option('-a', '--compute_fad', is_flag=True)
@click.option('-e', '--input_files', type=click.Path(exists=False), help="Path to input cvs file (if not specified, created from output_audio)")
@click.option('-z', '--stats', type=click.Path(exists=False), help="Path to use for stats file (if not specified, created from output_audio)")
def launcher(train, load, config, num_steps, num_workers, dataprocessor_type, sde,
             sampler_name, generate_num_samples, striding, input_audio, output_audio, 
             generate_batch_size, fs, guidance, diffusion_model_type, lr,
             batch_size, num_batches, ema_rate, load_ema, compute_fad,
             input_files, stats):
    
    c = click.get_current_context()
    c_params = c.params

    # Initiate CUDA process
    # Mandatory that there are available GPUs for training
    world_size = init_group_process(c_params["train"])

    # Load config as dict
    config_path = config
    config = load_yaml_config(config)
    
    # Update config dict with changed values from CLI
    for param in config.keys():
        if c_params[param] is not None:
            config[param] = c_params[param]
    config = AttrDict(config)
    config.world_size = world_size
    config.config_path = config_path

    # Generate smaller size configs for FAD and Generation purposes
    fad_config, generation_config = generate_configs(config)


    print(f'Using {config.world_size} GPUs')
    mp.spawn(main,
             args=(config, generation_config, fad_config),
             nprocs=config.world_size,
             join=True)


def main(config, gen_config, fad_config):

    if torch.cuda.is_available():
        dist.init_process_group(backend='nccl', world_size=config.world_size, rank=config.rank)
        torch.cuda.set_device(config.rank)
        config.device = f'cuda:{config.rank}'
    else:
        config.device = "cpu"

    # Loading the Handler, dealing with internal computation
    handler, model = load_handler(config)

    model_dir = None
    if config.load:
        model_dir = Path(config.config_path).parent
        handler.load(model_dir, load_ema=config.load_ema)

    # Run logger on main process
    run_logger = get_logger(config, model_dir)

    # == Training ==
    if config.train:
        handler.train_model(
            *config,
            run_logger=run_logger
        )
        sys.exit()

    model.eval()

    # == Handling saving of generated sample ==
    
    # Storing generated samples in subfolder
    if not Path('generated_samples/').exists():
        os.mkdir('generated_samples')

    import dataclasses
    gen_config = dataclasses.replace(
        gen_config,
        output_audio=(
            f'output/generated_samples/{config.sde}_'
            f'{config.diffusion_model_type}_{config.sampler_name}'
            f'_{config.timestamp}.wav'
        )
    )

    if Path(gen_config.output_audio).exists():
        warnings.warn(f'Destination file {gen_config.output_audio} exists, '
                      '(re)move it to generate audio; exiting')
        sys.exit()

    handler.generate_save(gen_config)

    # == Computing FAD ==

    if fad_config.compute_fad:
        fad_engine = get_fad_engine(gen_config, fad_config)
        fad_engine.compute_fad()


if __name__ == '__main__':
    launcher()
