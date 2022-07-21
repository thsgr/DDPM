import os
import json
import shutil
from pathlib import Path

# from copy import deepcopy
from datetime import datetime

from backup import make_repo_snapshot


def generate_run_name(config):
    # Compute time stamp if not present
    timestamp = config.get("timestamp")
    if not timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        config["timestamp"] = timestamp

    name_parts = [
        config["savename"],
        config["sde_kwargs"]["epsilon"],
        config["dataprocessor_type"],
    ]
    print(config)
    embedding = config["diffusion_model_kwargs"].get("embeddings")
    if embedding is not None:
        name_parts.append(embedding)
    name_parts.append(timestamp)
    run_name = "_".join(f"{part}" for part in name_parts)
    return run_name


class RunLogger:
    def _init(self, model_dir, config, load, world_size):
        if not load:
            config_path = config["config_path"]
            model_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(config_path, f"{model_dir}/config.py")
        # save some state that is not in the config file:
        # * world_size
        # * make a snapshot of the repo and store the snapshot name
        state = dict(world_size=world_size)
        if "repo_backup" in config:
            state["repo_snapshot"] = make_repo_snapshot(config["repo_backup"])
        with open(f"{model_dir}/state.json", "w") as f:
            json.dump(state, f)
        return state

    def log_epoch(self, *args):
        raise NotImplementedError


class TensorboardLogger(RunLogger):
    def __init__(self, config, load, world_size, model_dir=None):
        self.config = config
        self.load = load
        self.world_size = world_size
        if model_dir:
            self.model_dir = model_dir
        else:
            run_name = generate_run_name(config)
            base_dir = Path("models")
            self.model_dir = base_dir / run_name
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(os.fspath(self.model_dir))
        self._init(self.model_dir, config, load, world_size)

    def log_epoch(
        self, epoch_id, monitored_quantities_train, monitored_quantities_val=None
    ):
        loss_in_bins, mean_grad_norm = monitored_quantities_train
        self.writer.add_scalars("train/conditioned_loss", loss_in_bins, epoch_id)
        self.writer.add_scalar("train/mean_grad_norm", mean_grad_norm, epoch_id)
        if monitored_quantities_val:
            self.writer.add_scalars(
                "val/conditioned_loss", monitored_quantities_val, epoch_id
            )

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.writer.close()


class WandBLogger(RunLogger):
    def __init__(self, config, load, world_size, model_dir=None):
        import wandb

        debug = config["logging"].get("debug", False)
        if debug:
            print("Running in debug mode")
            self.run = wandb.init(mode="disabled")
        else:
            project = config["logging"].get("project", None)
            self.run = wandb.init(project=project)

        if model_dir:
            self.model_dir = model_dir
        else:
            base_dir = Path(config["logging"].get("logging_dir", "models"))
            self.model_dir = base_dir / self.run.name
        state = self._init(self.model_dir, config, load, world_size)
        hparams = preprocess_hparams(config, constructor_match, constructor_process)
        hparams.update(state)
        self.run.config.update(hparams)

    def log_epoch(
        self, epoch_id, monitored_quantities_train, monitored_quantities_val=None
    ):
        loss_in_bins, mean_grad_norm = monitored_quantities_train

        log_msg = {"mean_grad_norm": mean_grad_norm}
        for name, val in loss_in_bins.items():
            log_msg[f"train/{name}"] = val

        if monitored_quantities_val:
            loss_in_bins = monitored_quantities_val
            for name, val in loss_in_bins.items():
                log_msg[f"valid/{name}"] = val

        self.run.log(log_msg, step=epoch_id, commit=True)

    def __enter__(self):
        self.run.__enter__()

    def __exit__(self, *args):
        self.run.__exit__(*args)
        import wandb
        wandb.finish()


def constructor_match(key):
    return key == "constructor"


def constructor_process(val):
    return ":".join(val)


def preprocess_hparams(obj, pred, func):
    return (
        {
            k: preprocess_hparams(func(v) if pred(k) else v, pred, func)
            for k, v in obj.items()
        }
        if isinstance(obj, dict)
        else obj
    )
