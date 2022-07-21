from msilib.schema import Error
import sys
import importlib
import importlib
import json
from pathlib import Path
from typing import Dict, Any
import os
import yaml
import torch.distributed as dist
import torch 


def init_group_process(train):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(get_free_port())
    # === Set shared parameters

    # always use the maximum number of available GPUs for training
    # use CUDA_VISIBLE_DEVICES to set the GPUs to use
    if train:
        world_size = torch.cuda.device_count()
        if world_size == 0:
            raise ValueError("There needs to be GPUs to train the model")
    else:
        # only use 1 GPU for inference
        world_size = 1
    return world_size

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]
    
def is_main_process(device):
    if device == "cpu":
        return True
    return dist.get_rank() == 0

def dict_pretty_print(d, endstr='\n'):
    for key, value in d.items():
        if type(value) == list:
            print(f'{key.capitalize()}: [%s]' % ', '.join(map(str, value)))
        else:
            print(f'{key.capitalize()}: {value:.6}', end=endstr)

def display_monitored_quantities(epoch_id, monitored_quantities_train,
                                 monitored_quantities_val) -> None:
    if is_main_process():
        print(f'======= Epoch {epoch_id} =======')
        print(f'---Train---')
        dict_pretty_print(monitored_quantities_train, endstr=' ' * 5)
        print()
        print(f'---Val---')
        dict_pretty_print(monitored_quantities_val, endstr=' ' * 5)
        print('\n')
        
def all_reduce_scalar(scalar, average=True):
    t = torch.Tensor([scalar]).to(f'cuda:{dist.get_rank()}')
    dist.all_reduce(t)
    scalar = t[0].detach().item()
    if average:
        scalar = scalar / dist.get_world_size()
    return scalar

def cuda_variable(tensor, non_blocking=False):
    if torch.cuda.is_available():
        return tensor.to(f'cuda:{dist.get_rank()}', non_blocking=non_blocking)
    else:
        return tensor


def load_py_config(fp):
    modname = 'config'
    spec = importlib.util.spec_from_file_location(modname, fp)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module.config

def load_yaml_config(fp):
    with open(fp) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    data = dict(data)
    return data


def get_object_from_location(location: str, name: str) -> Any:
    module = importlib.import_module(location)
    return getattr(module, name)


def instantiate_from_dict(info: Dict[str, Any], defaults: Dict[str, Any] = {}) -> Any:
    kwargs = defaults.copy()
    kwargs.update(info["kwargs"])
    return get_object_from_location(*info["constructor"])(**kwargs)


def instantiate_from_path(fp: Path, defaults: Dict[str, Any] = {}) -> Any:
    with open(fp) as f:
        info = json.load(f)
    return instantiate_from_dict(info, defaults)


def rms(x):
    return (x**2).mean()**.5
