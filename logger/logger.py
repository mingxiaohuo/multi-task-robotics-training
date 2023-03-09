import logging
import logging.config
from pathlib import Path

from utils import read_json
#/apdcephfs/private_qinghonglin/video_codebase/frozen-in-time-main/logger/logger_config.json

def setup_logging(save_dir, log_config='/mnt/hdd1/ego4d_proj/mingxiaohuo_ego4d/EgoVLP/logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            #only for the key value, items
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
