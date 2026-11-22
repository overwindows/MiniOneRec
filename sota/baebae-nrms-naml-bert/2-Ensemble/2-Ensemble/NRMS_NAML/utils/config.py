######################################################################################################
# Mind 2020 competition
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: src/NRMS_NAML/utils/config.py
# - This file includes utility functions useful for reading configuration file
#
# Version: 1.0
#######################################################################################################

import yaml


def prepare_config(yaml_file, **kwargs):
    """
    Read a yaml file and return it as a dictionary.
    :param yaml_file: path of config file
    :param kwargs: additional configurations
    :return: dictionary
    """
    config = load_yaml(yaml_file)
    config = flat_config(config)
    config.update(kwargs)
    return config


def load_yaml(filename):
    """
    Load a yaml file.
    :param filename: path of the file
    :return: dictionary
    """
    try:
        with open(filename, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        return config
    except FileNotFoundError:  # for file not found
        raise
    except Exception as e:  # for other exceptions
        raise IOError("load {0} error!".format(filename))


def flat_config(config):
    """
    Flat config loaded from a yaml file to a flat dict.
    :param config: configuration dictionary
    :return: updated configuration dictionary
    """
    f_config = {}
    category = config.keys()
    for cate in category:
        for key, val in config[cate].items():
            f_config[key] = val
    return f_config
