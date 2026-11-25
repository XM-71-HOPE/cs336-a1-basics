from tokenizer.pretokenizer import PreTokenizer
from tokenizer.tokenizer import BPETokenizer, Tokenized
from utils import constants, logging_utils, io_utils
import logging
import time
from pathlib import Path

tokenizer = BPETokenizer()
tokenizer.read_tokenized_from_pkl("../results/tokenized_final.pkl")
tokenized = tokenizer.tokenized
assert type(tokenized) == Tokenized

for i in range(0, tokenized.size(), 200):
    (id, word) = (i, tokenized.vocab[i])
    print(id, end=": ")
    print(word)