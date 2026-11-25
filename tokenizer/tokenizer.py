from __future__ import annotations
from typing import NamedTuple
from multiprocessing import Pool
from utils import io_utils
from utils.constants import DEBUG, REGEX_PATTERN
from tokenizer.pretokenizer import PreTokenizer
from itertools import batched
from typing import Iterable
import os
import time
import logging


class BPETokenizer:
    def __init__(
        self: BPETokenizer,
    ):
        self.tokenized = None

    def naive_train(
        self: BPETokenizer,
        input_path: str,
        vocab_size: int = 10000,
        special_tokens: list[str] = ["<|endoftext|>"],
        pretokenize_pattern: str = REGEX_PATTERN,
        pretokenize_process_nums: int = 1,
        split_text_process_nums: int = 1,
        raw_text_split_num: int = 1,
        split_text_chunksize: int | None = None,
    ) -> Tokenized:
        """
        Tokenize the input .txt file
        returns the Tokenized object, also saves to self.tokenized
        currently the implementation after the pretokenization is single-threaded
        """
        if DEBUG.value == 'PROFILING':
            pretokenize_start = time.perf_counter()
        logging.info("Pretokenization started.")
        # special_tokens_encoded = [s.encode() for s in special_tokens]
        pretokenizer = PreTokenizer(
            pretokenize_pattern,
            special_tokens
        )
        pretokens = pretokenizer.pretokenize(
            input_path=input_path,
            process_nums_to_pretokenize = pretokenize_process_nums,
            process_nums_to_split_text = split_text_process_nums,
            raw_split_num=raw_text_split_num,
            split_nums_per_work=split_text_chunksize
        )
        logging.info("Pretokenization done.")

        if DEBUG.value == "PROFILING":
            pretokenize_end = time.perf_counter()
            logging.debug(f"time spent during pretokenization: {pretokenize_end - pretokenize_start}")

        self.tokenized = self.naive_train_pretokens(
            pretokens=pretokens,
            vocab_size = vocab_size,
        )

        return self.tokenized

    def naive_train_pretokens(
        self,
        pretokens: dict[tuple[bytes, ...], int],
        vocab_size: int = 10000,
        special_tokens: list[str] = ["<|endoftext|>"],
        # count_bp_process_nums: int = 1,
        # count_bp_batch_size: int = 1,
        # count_bp_chunksize: int = 256,
        # apply_merge_process_nums: int = 1
    ) -> Tokenized:
        """train from given pretokens, does not save the Tokenized object to self.tokenized"""
        special_tokens_encoded = [s.encode() for s in special_tokens]
        res = Tokenized(self._initialize_vocab(special_tokens_encoded=special_tokens_encoded), [])
        while res.size() < vocab_size:
            if DEBUG.value == 'PROFILING':
                round_start = time.perf_counter()
                logging.debug(f"round {res.size()+1} started")
            
            # bro you know what single process is actually way faster

            # _, max_bp = self._from_pretokens_to_bp_counts(
            #     pretokens=pretokens,
            #     process_nums=count_bp_process_nums,
            #     batch_size=count_bp_batch_size,
            #     chunksize=count_bp_chunksize
            # )

            _, max_bp = self._from_pretokens_to_bp_counts_single_with_max(
                pretokens = pretokens.items()
            )
            if DEBUG.value == 'PROFILING':
                bp_counted = time.perf_counter()
                logging.debug(f"time spent bp counting: {bp_counted - round_start}")
                
            res.merge.append(max_bp[0])
            res.vocab[res.size()]=max_bp[0].first + max_bp[0].second

            # pretokens = self._apply_merge_to_pretokens_naive(
            #     max_bp[0],
            #     pretokens,
            #     apply_merge_process_nums
            # )

            pretokens = dict(self._apply_merge_to_pretokens_naive_single(
                (max_bp[0], pretokens.items())
            ))
            if DEBUG.value == 'PROFILING':
                merged = time.perf_counter()
                logging.debug(f"time spent merging: {merged - bp_counted}")
            
        return res

    def read_tokenized_from_pkl(
        self,
        input_path: str,
    ) -> Tokenized:
        """read from the Tokenized .pkl file and write it into self.tokenized"""
        self.tokenized = io_utils.load_from_file(
            input_path
        )
        if type(self.tokenized) != Tokenized:
            raise TypeError(f"{input_path} is not a pickle of Tokenized")
        return self.tokenized

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
            for (bp, count) in self._from_pretoken_to_bp_counts(pt).items():
                if res.__contains__(bp):
                    res[bp] += count
                else:
                    res[bp] = count
        return res

    # def _from_pretokens_to_bp_counts_single_with_max(
    #     self,
    #     pretokens: Iterable[tuple[tuple[bytes, ...], int]]
    # ) -> tuple[dict[BytesPair, int], tuple[BytesPair, int]]:
    #     res = {}
    #     max_bp = (BytesPair(b'', b''), 0)
    #     for pt in pretokens:
    #         for (bp, count) in self._from_pretoken_to_bp_counts(pt).items():
    #             if res.__contains__(bp):
    #                 res[bp] += count
    #             else:
    #                 res[bp] = count
    #             if _greater((bp, res[bp]), max_bp):
    #                 max_bp = ((bp, res[bp]))
    #     return (res, max_bp)

    def _from_pretokens_to_bp_counts_single_with_max(
        self,
        pretokens: Iterable[tuple[tuple[bytes, ...], int]]
    ) -> tuple[dict[BytesPair, int], tuple[BytesPair, int]]:
        res = {}
        max_bp = (BytesPair(b'', b''), 0)
        for pt in pretokens:
            word, pt_count = pt
            L = len(word)
            for i in range(L - 1):
                bp = BytesPair(word[i], word[i+1])
                if res.__contains__(bp):
                    res[bp] += pt_count
                else:
                    res[bp] = pt_count
                if _greater((bp, res[bp]), max_bp):
                    max_bp = ((bp, res[bp]))
        return (res, max_bp)

    def _from_pretokens_to_bp_counts(
        self,
        pretokens: dict[tuple[bytes, ...], int],
        process_nums : int = 1,
        batch_size : int = 256,
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
        # batch_nums = (N - 1) // batch_size + 1     # ceil(N / batch_size)
        def _generate_tasks():
            return batched(
                iterable = pretokens.items(),
                n = batch_size
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
                    if _greater((bp, bp_dict[bp]), max_bp):
                        max_bp = (bp, bp_dict[bp])
        return (bp_dict, max_bp)

    def _apply_merge_to_pretoken_naive(
        self,
        params: tuple[BytesPair, tuple[tuple[bytes, ...], int]]
    ) -> tuple[tuple[bytes, ...], int]:
        """apply a merge to a pretoken"""
        bp_to_merge, pretoken = params
        word, count = pretoken
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == bp_to_merge:
                new_word.append(bp_to_merge.first + bp_to_merge.second)
                i+=2        # skips the merged bytes
                continue    
            new_word.append(word[i])
            i+=1
        
        return (tuple(new_word), count)

    def _apply_merge_to_pretokens_naive_single(
        self,
        params: tuple[BytesPair, Iterable[tuple[tuple[bytes, ...], int]]]
    ) -> Iterable[tuple[tuple[bytes, ...], int]]:
        bp_to_merge, pretokens = params
        res = []
        for pt in pretokens:
            res.append(self._apply_merge_to_pretoken_naive(
                (bp_to_merge, pt)
            ))
        return res

    def _apply_merge_to_pretokens_naive(
        self,
        bp_to_merge: BytesPair,
        pretokens: dict[tuple[bytes, ...], int],
        process_nums: int = 1
    ) -> dict[tuple[bytes, ...], int]:
        res = {}
        def _generate_tasks() -> list[tuple[BytesPair, Iterable[tuple[tuple[bytes, ...], int]]]]:
            tasks = []
            batch_size = (len(pretokens) - 1)//process_nums + 1     # ceil
            batches = batched(
                iterable = pretokens.items(),
                n = batch_size
            )
            for batch in batches:
                tasks.append((bp_to_merge, batch))
            return tasks
        tasks = _generate_tasks()
        with Pool(processes=process_nums) as pool:
            for new_pretokens in pool.imap_unordered(
                self._apply_merge_to_pretokens_naive_single,
                tasks
            ):
                for pt in new_pretokens:
                    res[pt[0]] = pt[1]
        return res

    def _initialize_vocab(
        self,
        special_tokens_encoded: list[bytes]
    ) -> dict[int, bytes]:
        vocab = {}
        for i in range(256):
            vocab[i] = bytes([i])

        for i in range(len(special_tokens_encoded)):
            vocab[i+256] = special_tokens_encoded[i]
        return vocab

class Tokenized(NamedTuple):
    vocab: dict[int, bytes]
    merge: list[BytesPair]
    def size(self) -> int:
        return len(self.vocab)

class BytesPair(NamedTuple):
    first: bytes
    second: bytes 

def _greater(
    a: tuple[BytesPair, int],
    b: tuple[BytesPair, int] 
) -> bool:
    bp_a, cnt_a = a
    bp_b, cnt_b = b
    return True if cnt_a > cnt_b else cnt_a == cnt_b and bp_a > bp_b