
from ConfigParser import SafeConfigParser
import ast

def get_config(config_file):
    config = SafeConfigParser()
    config.read(config_file)
    return config


def get_section_names(config):    
    return config.sections()


def has_section(config, section):
    return section in get_section_names(config)


def has_option(config, section, option):
    if has_section(config, section):
        sec = get_section_options(config, section)
    else:
        sec = []

    return option in sec


def get_section_options(config, section):
    dict1 = {}
    for k,v in config.items(section):
        try: 
            dict1[k] = ast.literal_eval(v)
        except ValueError:
            dict1[k] = v
    return dict1


def get_section_option(config, section, option):
    dict1 = get_section_options(config, section)
    return dict1[option]


