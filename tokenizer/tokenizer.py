from __future__ import annotations
from typing import NamedTuple
from multiprocessing import Pool
import os

class BPETokenizer:
    def __init__(
        self: BPETokenizer,
        vocab_size: int,
        special_tokens: list[str],
    ):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.special_tokens_encoded = [s.encode() for s in special_tokens]
    
    def tokenize(
        self: BPETokenizer,
        input_path: str
    ) -> Tokenized:
        """
        Tokenize the input .txt file
        """
        vocab = {}
        merge = []
        raise NotImplementedError

    def _split_text(self, input_path: str, process_nums: int = 1) -> list[tuple[int, int]]:
        """
        splits the input text, returns the byte position for each beginning of the special tokens in [st, ed) form
        note that this function assumes the heads are sparse and there will be hardly any collisions on lines
        """
        filesize = os.path.getsize(input_path)
        step_size = filesize // process_nums
        sts = range(0, filesize, step_size)[0:process_nums]
        sts = list(sts)
        sts.append(filesize)
        
        with Pool(processes=process_nums) as pool:
            res = pool.map      //TODO

        raise NotImplementedError
        
    def _find_eofs(self, st, ed, path) -> list[tuple[int, int]]:
        """[st, ed) format, returns the [st, ed) of each special token"""
        res = []
        fs = open(
            file = path,
            mode = "rb",
        )
        fs.seek(st)
        def _check_inline():
            if st == 0:
                return False
            fs.seek(st - 1)
            if fs.read() == b'\n':
                return False
        if _check_inline():
            fs.readline()   #skip the line in the middle, leave it to previous workers

        while fs.tell() < ed: 
            line_st = fs.tell()
            bline = fs.readline()
            tmp = []
            for special_token in self.special_tokens_encoded:
                found = 0
                while True:
                    found = bline.find(
                        special_token,
                        found
                    )
                    if found == -1:
                        break
                    tmp.append((found+line_st, found+special_token.__len__()+line_st))
            tmp.sort()
            res.extend(tmp)
        return res


class Tokenized(NamedTuple):
    vocab: dict[int, bytes]
    merge: list[tuple[bytes, bytes]]