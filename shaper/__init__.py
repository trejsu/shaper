import json
import logging.config

path = 'logging.json'
with open(path, 'rt') as f:
    config = json.load(f)
logging.config.dictConfig(config)
