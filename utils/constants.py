from enum import Enum

class _Debug(Enum):
    TOKENIZER = 'TOKENIZER'

DEBUG = _Debug.TOKENIZER
REGEX_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 