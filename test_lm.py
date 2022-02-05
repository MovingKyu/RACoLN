import torch
from src.util import *
from src.model import *
import logging
import os, sys, argparse
logging.basicConfig(
format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
datefmt="%Y-%m-%d %H:%M:%S",
level=os.environ.get("LOGLEVEL", "INFO").upper(),
stream=sys.stdout,
)
logger = logging.getLogger("test_lm.py")

parser = argparse.ArgumentParser(description='Argparse for Language Model')
parser.add_argument('--embedding-size', type=int, default=128,
                    help='yelp set to 128')
parser.add_argument('--hidden-size', type=int, default=500,
                    help='hidden size set to 500')
parser.add_argument('--batch-size', type=int, default=512,
                    help='batch size set to 512 for yelp')
parser.add_argument('--data', type=str, default="yelp",
                    help='data')
config = parser.parse_args()
if torch.cuda.is_available():
    config.device = "cuda"
else:
    config.device = "cpu"
config.data_path = f"data/{config.data}"

train, dev, test, train_iter, dev_iter, test_iter, X_VOCAB, C_LABEL = load_batch_iterator_with_eos(config.data_path, train="train.jsonl", val="dev.jsonl", test = "test.jsonl",
                        batch_size=config.batch_size,device=config.device)

bos_idx, pad_idx, eos_idx = getSpecialTokens(X_VOCAB)


lm = GRU_LM(len(X_VOCAB.vocab), config.embedding_size, config.hidden_size, pad_idx).to(config.device)
lm.load_state_dict(torch.load("model/{}_lm.pth".format(config.data)))
lm.eval()

# Test Set
test_iter = fetchIter(f"data/{config.data}/test.jsonl",X_VOCAB, C_LABEL, config.batch_size, config.device)

ce_val = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_idx)

ppl = computePPL(test_iter, lm, ce_val, bos_idx, pad_idx, eos_idx)
logger.info("[{}] PPL : {}".format("Test", ppl))