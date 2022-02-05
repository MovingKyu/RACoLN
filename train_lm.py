import pkbar
import math
from torch import nn, optim
import torch
import numpy as np
import torch.nn.functional as F
import math


from src.util import *
from src.model import *

class LmConfig:
    def __init__(self, data = "yelp", min_freq=4, batch_size=512, device="cuda", embedding_size = 128, hidden_size = 500, attn_size = 100, max_epoch = 20, seed = 0000, lr = 0.0005, eval_iter=100, max_iter=20000):
        self.data = data
        self.data_path = f"data/{data}/"
        self.min_freq = min_freq
        self.batch_size = batch_size
        self.device = device
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attn_size = attn_size 
        self.max_epoch = max_epoch
        self.seed = seed
        self.lr = lr
        self.pos_idx = 0
        self.neg_idx = 1
        self.log_iter = eval_iter
        self.max_iter = max_iter


import logging
import os
import sys
logging.basicConfig(
format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
datefmt="%Y-%m-%d %H:%M:%S",
level=os.environ.get("LOGLEVEL", "INFO").upper(),
stream=sys.stdout,
)
logger = logging.getLogger("train_lm.py")

config = LmConfig()
train, dev, test, train_iter, dev_iter, test_iter, X_VOCAB, C_LABEL = load_batch_iterator_with_eos(config.data_path, train="train.jsonl", val="dev.jsonl", test = "test.jsonl",
                        batch_size=config.batch_size,device=config.device)

input_size = len(X_VOCAB.vocab)
pad_idx = X_VOCAB.vocab.stoi["<pad>"]
bos_idx = X_VOCAB.vocab.stoi["<bos>"]
eos_idx = X_VOCAB.vocab.stoi["<eos>"]
lm = GRU_LM(len(X_VOCAB.vocab), config.embedding_size, config.hidden_size, pad_idx, dropout = 0.4).to(config.device)
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