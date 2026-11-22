from __future__ import annotations
from typing import NamedTuple
from multiprocessing import Pool
from utils.constants import DEBUG, REGEX_PATTERN
from tokenizer.pretokenizer import PreTokenizer
from itertools import batched
from typing import Iterable
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

    def train(self: BPETokenizer, input_path: str) -> Tokenized:
        """
        Tokenize the input .txt file
        """
        vocab = {}
        merge = []
        raise NotImplementedError

    def naive_train_pretokens(
        self,
        pretokens: dict[tuple[bytes, ...], int]
    ) -> Tokenized:
        raise NotImplementedError

    def _from_pretoken_to_bp_counts(
        self,
        pretoken: tuple[tuple[bytes, ...], int] 
    ) -> dict[BytesPair, int]:
        """converts the pretoken in tuple[tuple[bytes, ...], int] form into bytes pairs and its total counts"""
        word, pt_count = pretoken
        res = {}
        L = len(word)
        for i in range(L - 1):
            bp = BytesPair(word[i], word[i+1])
            if res.__contains__(bp):
                res[bp] += pt_count
            else:
                res[bp] = pt_count
        return res

    def _from_pretokens_to_bp_counts_single(
        self,
        pretokens: Iterable[tuple[tuple[bytes, ...], int]]
    ) -> dict[BytesPair, int]:
        res = {}
        for pt in pretokens:
            for (bp, count) in self._from_pretoken_to_bp_counts(pt):
                if res.__contains__(bp):
                    res[bp] += count
                else:
                    res[bp] = count
        return res

    def _from_pretokens_to_bp_counts(
        self,
        pretokens: dict[tuple[bytes, ...], int],
        process_nums : int = 1,
        batch_size : int = 1,
        chunksize: int | None = None
    ) -> tuple[dict[BytesPair, int], tuple[BytesPair, int]]:
        """
        computes the bp counts for a pretoken dictionary

        Args:
            pretokens: the pretoken dictionary to work on
            process_nums: the number of workers to process the pretokens in parallel
            batch_size: the number of pretokens in one task,
            chunksize: the number of batches per work
        
        Returns:
            a tuple of:
            dict[BytesPair, int]: a dict from bytes pair to the counts of occurrence
            tuple[BytesPair, int]: the BytesPair to merge
        """
        bp_dict = {}
        max_bp = (BytesPair(b'', b''), 0)
        N = len(pretokens)
        batch_nums = (N - 1) // batch_size + 1     # ceil(N / batch_size)
        def _generate_tasks():
            return batched(
                iterable = pretokens.items(),
                n = batch_nums
            )
        tasks = _generate_tasks()
        with Pool(processes = process_nums) as pool:
            for task_count_dict in pool.imap_unordered(
                self._from_pretokens_to_bp_counts_single,
                tasks,
                chunksize=chunksize
            ):
                for (bp, count) in task_count_dict.items():
                    if bp_dict.__contains__(bp):
                        bp_dict[bp] += count
                    else:
                        bp_dict[bp] = count
                    if _greater(bp_dict[bp], max_bp):
                        max_bp = bp_dict[bp]
        return (bp_dict, max_bp)

    
    

class Tokenized(NamedTuple):
    vocab: dict[int, bytes]
    merge: list[BytesPair]

class BytesPair(NamedTuple):
    first: bytes
    second: bytes 

def _greater(
    a: tuple[BytesPair, int],
    b: tuple[BytesPair, int] 
) -> bool:
    bp_a, cnt_a = a
    bp_b, cnt_b = b
    return True if cnt_a > cnt_b else bp_a > bp_b