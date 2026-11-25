from __future__ import annotations
import logging
from multiprocessing import Pool
import os
import regex as re

from utils.constants import DEBUG

# class for pretokenization
# inputs a string and pretokenize it according to a regex
class PreTokenizer:
    def __init__(
        self: PreTokenizer,
        pattern: str,
        special_tokens: list[str]
    ):
        """
        Initializes a PreTokenizer

        Args:
            pattern: a regex defining the pattern to pre-tokenize
            special_tokens: a list of strings that splits the document and gets excluded during pretokenization
        """
        self.pattern = pattern
        self.special_tokens = special_tokens
        self.special_tokens_encoded = [s.encode('utf-8') for s in special_tokens]

    def pretokenize(
        self,
        input_path: str,
        process_nums_to_pretokenize: int = 1,
        process_nums_to_split_text: int = 1,
        raw_split_num: int = 1,
        split_nums_per_work: int | None = None
    ) -> dict[tuple[bytes, ...], int]:
        return self._pretokenize_splits(
            input_path=input_path,
            process_nums_to_pretokenize=process_nums_to_pretokenize,
            splits = self._split_text(
                input_path=input_path,
                raw_split_num=raw_split_num,
                process_nums = process_nums_to_split_text,
                split_nums_per_work=split_nums_per_work
            )
        )
        raise NotImplementedError

    def _pretokenize_splits(
            self, 
            input_path: str,
            splits: list[tuple[int, int]],
            process_nums_to_pretokenize: int = 1,
                     ) -> dict[tuple[bytes, ...], int]:
        """pretokenizes the input files with splits, returns an dict on 'word counts'"""

        def _generate_tasks() -> list[tuple[str, tuple[int, int]]]:
            res = []
            for tp in splits:
                res.append(
                    (input_path, tp)
                )
            return res

        tasks = _generate_tasks()

        with Pool(process_nums_to_pretokenize) as pool:
            res = {}
            for pretokens in pool.imap_unordered(
                self._pretokenize_single_split,
                tasks
            ):
                for item in pretokens.items():
                    if res.__contains__(item[0]):
                        res[item[0]] += item[1]
                        continue
                    res[item[0]] = item[1]
        return res

    def _pretokenize_single_sentence(
        self,
        input: str,
    ) -> dict[tuple[bytes, ...], int]:
        """
        Pre-tokenizes a raw sentence according to self.pattern

        Args:
            input: the input string to pre-tokenize
        
        Returns:
            dict[tuple[bytes, ...], int] representing the pre-tokens in the tuple of the characters and the word count
        """
        res = {}
        iter = re.finditer(
            self.pattern,
            input,
        )
        for _match in iter:
            # chunk = _match.group()
            chunk = _match.group().encode()
            chars = []
            for i in range(len(chunk)):
                # chars.append(chunk[i].encode("utf-8"))
                chars.append(chunk[i:i+1])
                pass
            chars = tuple(chars)

            if res.__contains__(chars):
                res[chars] += 1
            else:
                res[chars] = 1
        return res

    def _pretokenize_single_split(
        self,
        params: tuple[str, tuple[int, int]]
    ) -> dict[tuple[bytes, ...], int]:
        file, split = params 
        st, ed = split
        if st<0:
            raise ValueError
        if ed<=st:
            raise ValueError
        with open(file=file, mode='rb') as fs:
            fs.seek(st)
            text = fs.read(ed - st)
            res = self._pretokenize_single_sentence(
                text.decode('utf-8')
            )
        return res


    def _from_eofs_to_splits(
        self,
        filesize: int,
        eofs: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """
        transforms the eof tags of list[[st, ed)] into text splits of list[[st, ed)]
        special tokens (eofs) are not included
        this function assumes eofs are ordered
        """
        res = []
        N = eofs.__len__()
        if N == 0:
            res.append((0, filesize))
            return res
        if eofs[0][0] != 0:         # prevent empty strings
            res.append((0, eofs[0][0]))
        for i in range(N - 1):
            if eofs[i][1] != eofs[i+1][0]:  # prevent empty strings
                res.append((eofs[i][1], eofs[i+1][0]))
        if eofs[N-1][1] != filesize:
            res.append((eofs[N-1][1], filesize))
        return res

    def _find_eofs_complete(
        self, input_path: str, raw_split_num: int = 1, process_nums: int = 1, split_nums_per_work: int | None = None
    ) -> list[tuple[int, int]]:
        """
        finds all the special tokens in the input text, returns the byte position for each beginning of the special tokens in [st, ed) form
        note that this function assumes the splits are sparse and there will be hardly any collisions on lines

        Args:
            input_path: the path to the .txx file to tokenize
            raw_split_num: the number of chunks you divide the input text for processes to split in parallel
            process_nums: the number of processes
            split_nums_per_work: the number of chunks for a worker to work on

        Returns:
            list[tuple[int, int]]: special tokens in the form of [st, ed) byte position
        """
        filesize = os.path.getsize(input_path)

        #define the starting point of each process
        step_size = filesize // raw_split_num
        sts = range(0, filesize, step_size)[0:raw_split_num]
        sts = list(sts)
        sts.append(filesize)
        final_result = []

        # generate tasks for splitting
        def _generate_tasks() -> list[tuple[int, int, str]]:
            res = []
            for i in range(sts.__len__() - 1):
                res.append((sts[i], sts[i + 1], input_path))
            return res

        tasks = _generate_tasks()
        with Pool(processes=process_nums) as pool:
            res = pool.starmap(self._find_eofs_single, tasks, split_nums_per_work)
        for chunk in res:
            final_result.extend(chunk)

        if DEBUG.value == "PRETOKENIZER":
            # print(final_result)
            logging.debug(final_result)
        return final_result

    def _find_eofs_single(self, st, ed, path) -> list[tuple[int, int]]:
        """[st, ed) format, returns the [st, ed) of each special token"""
        res = []
        fs = open(
            file=path,
            mode="rb",
        )
        fs.seek(st)

        def _check_inline():
            if st == 0:
                return False
            fs.seek(st - 1)
            if fs.read(1) == b"\n":
                return False

        if _check_inline():
            fs.readline()  # skip the line in the middle, leave it to previous workers

        while fs.tell() < ed:
            line_st = fs.tell()
            bline = fs.readline()
            tmp = []
            for special_token in self.special_tokens_encoded:
                found = 0
                while True:
                    found = bline.find(special_token, found)
                    if found == -1:
                        break
                    tmp.append(
                        (found + line_st, found + special_token.__len__() + line_st)
                    )
                    found += special_token.__len__()
            tmp.sort()
            res.extend(tmp)
        return res

    def _split_text(self, input_path: str, raw_split_num: int = 1, process_nums: int = 1, split_nums_per_work: int | None = None) -> list[tuple[int, int]]:
        """split the input text into [st, ed) form text chunks, excluding all the special tokens"""
        filesize = os.path.getsize(input_path)
        return self._from_eofs_to_splits(
            filesize=filesize,
            eofs = self._find_eofs_complete(
                input_path=input_path,
                raw_split_num=raw_split_num,
                process_nums=process_nums,
                split_nums_per_work=split_nums_per_work
            )
        )
