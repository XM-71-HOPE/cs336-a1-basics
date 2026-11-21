from __future__ import annotations
from typing import NamedTuple
from multiprocessing import Pool
from utils.constants import DEBUG, REGEX_PATTERN
from tokenizer.pretokenizer import PreTokenizer
import os


class BPETokenizer:
    def __init__(
        self: BPETokenizer,
        vocab_size: int,
        special_tokens: list[str],
        pre_tokenize_pattern: str = REGEX_PATTERN
    ):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.special_tokens_encoded = [s.encode() for s in special_tokens]
        self.pretokenizer = PreTokenizer(
            pattern=pre_tokenize_pattern,
            special_tokens=special_tokens
        )

    def tokenize(self: BPETokenizer, input_path: str) -> Tokenized:
        """
        Tokenize the input .txt file
        """
        vocab = {}
        merge = []
        raise NotImplementedError



class Tokenized(NamedTuple):
    vocab: dict[int, bytes]
    merge: list[tuple[bytes, bytes]]
