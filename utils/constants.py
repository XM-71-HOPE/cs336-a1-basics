from enum import Enum

class _Debug(Enum):
    PRETOKENIZER = 'PRETOKENIZER'
    TOKENIZER = 'TOKENIZER'
    PROFILING = 'PROFILING'

DEBUG = _Debug.PROFILING
REGEX_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 
LOG_PATH_RELATIVE = "./log/"