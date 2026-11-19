from tokenizer.pre_tokenizer import PreTokenizer

REGEX_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 

preTokenizer = PreTokenizer(REGEX_PATTERN)

preTokens = preTokenizer.compute(
    """Violet, this is XyloMeta. I've successfully infiltrated the frontier, over.
    XyloMeta, this is Violet. I'll catch up with you 23 minutes later, at zeta33minimal, over."""
)

print(preTokens)
