import logging
from datetime import datetime
from constants import LOG_PATH_RELATIVE
import time


def get_current_log_filename(
    filename_prefix: str,
    prepend_log_path: bool = False
) -> str:
    res = ''
    if prepend_log_path:
        res += LOG_PATH_RELATIVE
    now = datetime.now()
    timestr = now.strftime(
        format="%m%d%H%M%S"
    )
    res += filename_prefix+timestr+'.log'
    return res
