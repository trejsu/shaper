import json
import logging.config

from config import LOG_CONFIG

with open(LOG_CONFIG, 'rt') as f:
    config = json.load(f)
logging.config.dictConfig(config)
