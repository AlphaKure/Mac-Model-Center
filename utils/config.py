import configparser

CONFIGPATH = "project.ini"

config = configparser.ConfigParser()


def read_config(section: str, key: str):
    """Read config"""

    global config
    config.read(CONFIGPATH)
    try:
        return config[section][key]
    except:
        return None