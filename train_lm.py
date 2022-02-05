import math
import torch
import logging
import os
import sys
from src.util import *
from src.model import *
import argparse
import numpy as np
import random

logging.basicConfig(
format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
datefmt="%Y-%m-%d %H:%M:%S",
level=os.environ.get("LOGLEVEL", "INFO").upper(),
stream=sys.stdout,
)
logger = logging.getLogger("train_lm.py")


parser = argparse.ArgumentParser(description='Argparse for Language Model')
parser.add_argument('--embedding-size', type=int, default=128,
                    help='yelp set to 128')
parser.add_argument('--hidden-size', type=int, default=500,
                    help='hidden size set to 500')
parser.add_argument('--batch-size', type=int, default=512,
                    help='batch size set to 512 for yelp')
parser.add_argument('--data', type=str, default="yelp",
                    help='data')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--seed', type=int, default=0000,
                    help='random seed')
parser.add_argument('--dropout-p', type=float, default=0.4,
                    help='dropout probability')
parser.add_argument('--max-epoch', type=int, default=20,
                    help='max epoch')
parser.add_argument('--log_iter', type=int, default=100,
                    help='log iteration')
config = parser.parse_args()
if torch.cuda.is_available():
    config.device = "cuda"
else:
    config.device = "cpu"
config.data_path = f"data/{config.data}"

torch.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)

train, dev, test, train_iter, dev_iter, test_iter, X_VOCAB, C_LABEL = load_batch_iterator_with_eos(config.data_path, train="train.jsonl", val="dev.jsonl", test = "test.jsonl",
                        batch_size=config.batch_size,device=config.device)

bos_idx, pad_idx, eos_idx = getSpecialTokens(X_VOCAB)
lm = GRU_LM(len(X_VOCAB.vocab), config.embedding_size, config.hidden_size, pad_idx, dropout = config.dropout_p).to(config.device)
lm_optim = torch.optim.Adam(lm.parameters(), config.lr)
ce = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
ce_val = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_idx)

min_ppl = float("inf")
for epoch in range(config.max_epoch):
    for i, batch in enumerate(train_iter):
        lm_optim.zero_grad()
        lm_text = prepareBatchForLM(batch.X[0], bos_idx, pad_idx, eos_idx)
        logit = lm(lm_text.T, batch.X[1])
        logit = logit.view(-1, logit.size(2))
        target = torch.reshape(batch.X[0].T, (-1,))
        loss = ce(logit, target)
        loss.backward()
        lm_optim.step()
        if i%config.log_iter==0:
            logger.info("Train Epoch {}-{} PPL : {}".format(epoch, i, loss.exp().item()))
    lm.eval()
    ppl = []
    length=0
    for i, batch in enumerate(dev_iter):
        lm_text = prepareBatchForLM(batch.X[0], bos_idx, pad_idx, eos_idx)
        logit = lm(lm_text.T, batch.X[1])
        logit = logit.view(-1, logit.size(2))
        target = torch.reshape(batch.X[0].T, (-1,))
        ppl.extend(ce_val(logit, target).tolist())
        length+=batch.X[1].sum()
    val_loss = sum(ppl)/length
    val_ppl = math.exp(val_loss)
    logger.info("Validation Epoch {} PPL : {}".format(epoch, val_ppl))
    if val_ppl<min_ppl:
        min_ppl = val_ppl
        torch.save(lm.state_dict(), f"model/{config.data}_lm.pth")
        logger.info("Epoch {} Checkpoint Saved with PPL {}".format(epoch, val_ppl))
    lm.train()