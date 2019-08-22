import logging
import os
from logging import handlers
from fa_config import fa_config as fc

LOG_FORMAT = '%(asctime)s %(levelname)s "%(filename)s" line:%(lineno)d: %(message)s'
# LOG_FORMAT = '%(asctime)s %(levelname)s - %(message)s'
DATE_FORMAT = '%m-%d %H:%M:%S '
# save_result = os.path.join(os.path.dirname(__file__), 'log')
save_result = os.path.join(fc.root_path, 'log')
#print(save_result)
if not os.path.exists(save_result):
    os.mkdir(save_result)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# logging.basicConfig(level = logging.DEBUG, format = LOG_FORMAT, datefmt = DATE_FORMAT)

#the file to write the log info
s_handle = logging.StreamHandler()
rf_handle = logging.FileHandler(save_result + '/eighth_resmobile_train.log')

#segment the log file by the max bytes.
# rf_handle = logging.handlers.RotatingFileHandler(filename = save_result + '/err.log',
#                                                     maxBytes = 50*1024*1024, backupCount = 10)

s_handle.setLevel(logging.DEBUG)
s_handle.setFormatter(logging.Formatter(LOG_FORMAT))
rf_handle.setLevel(logging.DEBUG)
rf_handle.setFormatter(logging.Formatter(LOG_FORMAT))

logger.addHandler(s_handle)
logger.addHandler(rf_handle)
