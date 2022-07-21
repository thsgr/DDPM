import dataclasses

@dataclasses.dataclass
class GenerationConfig:
    input_audio: str
    output_audio: str
    num_samples: int
    batch_size: int
    fs: int = 16000
    guidance: float = 1.0

@dataclasses.dataclass
class FADConfig:
    compute_fad: bool
    input_files: str
    stats: str
    background_stats: str