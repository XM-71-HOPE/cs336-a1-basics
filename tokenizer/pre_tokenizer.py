from __future__ import annotations
import regex as re

# class for pretokenization
# inputs a string and pretokenize it according to a regex
class PreTokenizer:
    def __init__(
        self: PreTokenizer,
        pattern: str,
    ):
        """
        Initializes a PreTokenizer

        Args:
            pattern: a regex defining the pattern to pre-tokenize
        """
        self.pattern = pattern

    def compute(
        self,
        input: str,
    ) -> dict[tuple[bytes, ...], int]:
        """
        Pre-tokenizes the input string

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
            chunk = _match.group()
            chars = []
            for i in range(len(chunk)):
                chars.append(chunk[i:i+1].encode("utf-8"))
            chars = tuple(chars)

            if res.__contains__(chars):
                res[chars] += 1
            else:
                res[chars] = 1
        return res

