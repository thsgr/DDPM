from .fad_engine import FAD


def get_fad_engine(gen_config, fad_config)->FAD:
    return FAD(gen_config, fad_config)