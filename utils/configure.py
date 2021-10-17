import codecs
from collections import namedtuple

import yaml


def load_config(config_path: str) -> dict:
    with codecs.open(config_path, "r", encoding="utf-8") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # convert dict to namedtuple
    params = namedtuple("Params", params.keys())(**params)
    return params
