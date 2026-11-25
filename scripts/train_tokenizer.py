from tokenizer.pretokenizer import PreTokenizer
from tokenizer.tokenizer import BPETokenizer
from utils import constants, logging_utils, io_utils
import logging
import time
from pathlib import Path

log_path = Path(constants.LOG_PATH_RELATIVE)
if not log_path.exists():
    log_path.mkdir()

logging.basicConfig(
    filename=logging_utils.get_current_log_filename('run-', prepend_log_path=True),
    level=logging.DEBUG,
    format="[%(levelname)s] %(asctime)s %(message)s",
)

tokenizer = BPETokenizer(
)

input_path = "../data/TinyStoriesV2-GPT4-valid.txt"
# input_path = "../data/TinyStoriesV2-GPT4-train.txt"
# input_path = "../data/tmp.txt"

tokenized = tokenizer.naive_train(
    input_path=input_path,
    vocab_size = 1600,
    special_tokens = ['<|endoftext|>'],
    pretokenize_pattern=constants.REGEX_PATTERN,
    pretokenize_process_nums=16,
    raw_text_split_num=64,
    split_text_chunksize=4,
)

io_utils.serialize_to_file(
    filename="../results/tokenized_test.pkl",
    object=tokenized,
    force=False
)

for i in range(0, tokenized.size(), 200):
    (id, word) = (i, tokenized.vocab[i])
    print(id, end=": ")
    print(word)