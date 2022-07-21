import subprocess
import json
from pathlib import Path
from datetime import datetime
import logging

"""
Create snapshots of the (dirty) repo worktree with a single
function call (`make_repo_snapshot`). This function can be called at
the start of the train run to improve reproducibility. It relies on
the `borg` executable, which can be installed via pip.

See: https://borgbackup.readthedocs.io/en/stable/

"""


def make_repo_snapshot(config, fp=None):
    """Make a borg snapshot of the repo and return the snapshot name.
    When a filepath `fp` is provided the name is written to the file.

    """
    if not have_borg():
        raise Exception(
            "borg not installed, cannot make repo snapshot "
            "(install borgbackup package)"
        )
    ensure_repo(config)
    snapshot_name = run_backup(config)
    if fp is not None:
        with open(fp, "w") as f:
            f.write(snapshot_name)
    return snapshot_name

def have_borg():
    try:
        import borg
    except ImportError:
        return False
    return True


def init_repo(backup_location):
    cmd = [
        "borg",
        "init",
        "-e",
        "none",
        "--log-json",
        "--make-parent-dirs",
        backup_location,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode:
        raise Exception(f"borg repo init failed (cmd={' '.join(cmd)})")


def ensure_repo(config):
    backup_location = config.get("backup_location")
    if backup_location:
        if not Path(backup_location).exists():
            init_repo(backup_location)
            logger = logging.getLogger(__name__)
            logger.info(f'creating repo at "{backup_location}"')
    else:
        raise Exception('"backup_location" key missing in "repo_backup" config')


def run_backup(config):
    backup_location = config["backup_location"]
    archive_name = datetime.strftime(datetime.now(), "%Y-%m-%d--%H:%M:%S")
    target = f"{backup_location}::{archive_name}"
    source = config.get("source_dir", str(Path.cwd()))
    cmd = [
        "borg",
        "create",
        "--log-json",
        "--one-file-system",
        "--compression",
        "zstd",
        "--noatime",
    ]
    patterns = config.get("pattern_file")
    if patterns:
        cmd.extend(["--patterns-from", patterns])
    cmd.append(target)
    cmd.append(source)
    logger = logging.getLogger(__name__)
    logger.info(f"creating snapshot of {source} at {target}")
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode:
        stdout = result.stdout.decode("utf8")
        stderr = result.stderr.decode("utf8")
        stderr_msg = json.loads(stderr)
        
        raise Exception(
            f"borg archive creation failed with message: \"{stderr_msg.get('message')}\" (cmd: {' '.join(cmd)})"
        )
    return archive_name
