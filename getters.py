from configs.misc_configs import FADConfig, GenerationConfig
from diffusion.run_logging import TensorboardLogger, WandBLogger
from handlers.diffusion_handler import DDPMHandler
from handlers.handler import Handler
from diffusion.samplers.sampler import EMSampler, Sampler
from diffusion.sdes.sde import DDPMSde
from data.parametrization import DataParametrization

from data.dataloaders import DataloaderGenerator, AudioDataloaderGenerator
from data.dataprocessors import DataProcessor, MuLawDataProcessor
from models import UNet
from diffusion.sdes import SDE
from diffusion.utils import instantiate_from_dict, is_main_process
from torch.nn.parallel import DistributedDataParallel


def get_dataloader_generator(
        dataset: str,
        dataloader_generator_kwargs: dict,
        rank=0,
        world_size=1) -> DataloaderGenerator:
    """[summary]
    Args:
        dataset (str): [description]
        dataloader_generator_kwargs (dict): [description]
    Returns:
        DataloaderGenerator: [description]
    """
    if dataset is not None:
        default_kwargs = dict(rank=rank, world_size=world_size)
        return instantiate_from_dict(dataloader_generator_kwargs, default_kwargs)
    else:
        raise NotImplementedError("Please add dataset to datasets list in get_dataloader function")


def get_dataprocessor(dataprocessor_type: str) -> DataProcessor:
    """[summary]
    Args:
        dataprocessor_type (str): [description]
        dataprocessor_kwargs (dict): [description]
    Returns:
        DataProcessor: [description]
    """
    if dataprocessor_type == None :        
        return DataProcessor()
    elif dataprocessor_type == 'mu-law':
        return MuLawDataProcessor()
    else:
        raise NotImplementedError


def get_diffusion_model(diffusion_model_type: str,
                        diffusion_model_kwargs: dict = None,
                        dataloader_generator: DataloaderGenerator = None,
                        ):
    if diffusion_model_type == 'u-net':
        assert isinstance(dataloader_generator, AudioDataloaderGenerator)
        return UNet()
    elif diffusion_model_type == "custom":
        return instantiate_from_dict(diffusion_model_kwargs)
    else:
        raise NotImplementedError

def get_handler(model,
                dataloader_generator,
                data_processor,
                sde,
                sampler,
                parametrization,
                config, ) -> Handler:

    sdes = ['ddpm']

    if config.sde in sdes:

        return DDPMHandler(
            model=model,
            dataloader_generator=dataloader_generator,
            data_processor = data_processor,
            sde=sde,
            sampler=sampler,
            ema_rate=config.ema_rate,
            parametrization=parametrization,
        )

    else:
        raise NotImplementedError("Please choose a correct handler")


def get_parametrization(model_parametrization: dict = None) -> DataParametrization:
    
    if model_parametrization is None:
        return None
    else:
        raise NotImplementedError


def get_sde(sde_name: str) -> SDE:
    
    if sde_name == 'ddpm':
        return DDPMSde()
    else:
        raise NotImplementedError


def get_sampler(sampler_name, num_steps,
                sde, model, parametrization) -> Sampler:

        if sampler_name == 'EM':
            return EMSampler(num_steps, sde, model, parametrization)
        else:
            raise NotImplementedError


def get_logger(config, model_dir):
    if is_main_process(config.device) and config.train:
        
        logging_config = config.get('logging', {})
        if logging_config.get('type') == 'wandb':
            run_logger = WandBLogger(config, config.load, 
                                     config.world_size, model_dir)
        else:
            run_logger = TensorboardLogger(config, config.load, 
                                           config.world_size, model_dir)
    else:
        run_logger = None
    
    return run_logger


def load_handler(config):
    # dataloader generator
    dataloader_generator_kwargs = {
        "train_paths": config.train_paths,
        "val_paths": config.val_paths,
        "test_paths": config.test_paths,
        "audio_length":config.audio_length,
    }

    dataloader_generator: DataloaderGenerator = get_dataloader_generator(
        dataset=config.dataset,
        dataloader_generator_kwargs=dataloader_generator_kwargs,
        rank=config.rank, world_size=config.world_size)

    # data processor
    data_processor: DataProcessor = get_dataprocessor(
        dataloader_generator=dataloader_generator,
        dataprocessor_type=config.dataprocessor_type,
    )

    # TODO : Add kwargs here from config if there are any necessary
    diffusion_model_kwargs = {

    }
    model = get_diffusion_model(
        diffusion_model_type=config.diffusion_model_type,
        diffusion_model_kwargs=diffusion_model_kwargs,
        dataloader_generator=dataloader_generator,
        dataprocessor=data_processor,
    )

    model.to(config.device)

    model = DistributedDataParallel(module=model,
                                    device_ids=[config.rank],
                                    output_device=config.rank)

    model_parametrization = config.get('model_parametrization')
    parametrization = get_parametrization(
        model_parametrization=model_parametrization
        )

    sde = get_sde(
        sde_name=config.sde,
    )

    sampler = get_sampler(
        config.sampler_name, config.num_steps,
        sde, model, parametrization,
    )

    handler = get_handler(
        model=model,
        dataloader_generator=dataloader_generator,
        data_processor = data_processor,
        sde=sde,
        sampler=sampler,
        parametrization=parametrization,
        config=config,
        )
    
    return handler, model

def generate_configs(config):

    if config.compute_fad:
        background_stats = f"{config.fad_dir_path}stats/{config.dataset}_stats"
    else:
        background_stats = ''
    fad_config = FADConfig(
        config.compute_fad,
        config.input_files,
        config.stats,
        background_stats,
    )

    generation_config = GenerationConfig(
        config.input_audio,
        config.output_audio,
        config.generate_num_samples,
        config.generate_batch_size,
        guidance=config.guidance
    )

    return fad_config, generation_config