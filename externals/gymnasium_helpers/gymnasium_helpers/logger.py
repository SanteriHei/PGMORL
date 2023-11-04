import logging
import logging.config
import pathlib

from ruamel.yaml import YAML

_CONFIG_SET: bool = False
_CONFIG_PATH: pathlib.Path = (
        pathlib.Path(__file__).resolve().parents[3] / "configs" / "logging_config.yml"
)


def get_logger(logger_name: str) -> logging.Logger:
    _maybe_set_config(_CONFIG_PATH)
    return logging.getLogger(logger_name)

def _maybe_set_config(path: str | pathlib.Path):
    '''
    Set the configuration for the logging lib if not set already.

    Parameters
    ----------
    path: str | pathlib.Path:
        The path to the logging configuration
    '''
    global _CONFIG_SET
    if _CONFIG_SET:
        return

    path = pathlib.Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError((f"{str(path)!r} does not point to a valid "
                                 "logging configuration file!"))

    yaml = YAML(typ="safe")
    with path.open("r") as ifstream:
        conf = yaml.load(ifstream)
    logging.config.dictConfig(conf)
    _CONFIG_SET = True


