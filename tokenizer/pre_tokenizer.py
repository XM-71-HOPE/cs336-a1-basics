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
    ) -> dict[bytes, int]:
        """
        Pre-tokenizes the input string

        Args:
            input: the input string to pre-tokenize
        
        Returns:
            dict[bytes, int] representing the pre-tokens and the word count
        """
        res = {}
        iter = re.finditer(
            self.pattern,
            input,
        )
        for _match in iter:
            chunk = _match.group()
            if res.__contains__(chunk):
                res[chunk] += 1
            else:
                res[chunk] = 1
        return res

        
    