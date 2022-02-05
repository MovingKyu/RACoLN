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
lm.load_state_dict(torch.load("model/{}_lm.pth".format(config.data)))



# output= TabularDataset(path="output/reverseAttention.jsonl",
# output= TabularDataset(path="output/styleTransformer.jsonl",
output= TabularDataset(path="output/CROSSALIGNED.jsonl",
                  format="json",
                  fields={"X":("X", X_VOCAB),
                          "C": ('C', C_LABEL)})
output_iter = BucketIterator(output, batch_size=config.batch_size,sort_key=lambda x: len(x.X), device=config.device, shuffle=False)

refs= TabularDataset(path=f"data/{config.data}/sentiment.ref.jsonl",
                  format="json",
                  fields={"X":("X", X_VOCAB),
                          "C": ('C', C_LABEL)})

ref_iter = BucketIterator(refs, batch_size=config.batch_size,sort_key=lambda x: len(x.X), device=config.device, shuffle=False)

test= TabularDataset(path=f"data/{config.data}/test.jsonl",
                  format="json",
                  fields={"X":("X", X_VOCAB),
                          "C": ('C', C_LABEL)})

test_iter = BucketIterator(test, batch_size=config.batch_size,sort_key=lambda x: len(x.X), device=config.device, shuffle=False)

ce_val = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_idx)
lm.eval()

ppl = []
length = 0
for i, batch in enumerate(test_iter):
    lm_text = prepareBatchForLM(batch.X[0], bos_idx, pad_idx, eos_idx)
    logit = lm(lm_text.T, batch.X[1])
    logit = logit.view(-1, logit.size(2))
    target = torch.reshape(batch.X[0].T, (-1,))
    ppl.extend(ce_val(logit, target).tolist())
    length += batch.X[1].sum()
val_loss = sum(ppl)/length
val_ppl = math.exp(val_loss)
logger.info("[{}] PPL : {}".format("Test", val_ppl))

ppl = []
length = 0
for i, batch in enumerate(ref_iter):
    lm_text = prepareBatchForLM(batch.X[0], bos_idx, pad_idx, eos_idx)
    logit = lm(lm_text.T, batch.X[1])
    logit = logit.view(-1, logit.size(2))
    target = torch.reshape(batch.X[0].T, (-1,))
    ppl.extend(ce_val(logit, target).tolist())
    length += batch.X[1].sum()
val_loss = sum(ppl)/length
val_ppl = math.exp(val_loss)
logger.info("[{}] PPL : {}".format("Ref", val_ppl))

ppl = []
length = 0
for i, batch in enumerate(output_iter):
    lm_text = prepareBatchForLM(batch.X[0], bos_idx, pad_idx, eos_idx)
    logit = lm(lm_text.T, batch.X[1])
    logit = logit.view(-1, logit.size(2))
    target = torch.reshape(batch.X[0].T, (-1,))
    ppl.extend(ce_val(logit, target).tolist())
    length += batch.X[1].sum()
val_loss = sum(ppl)/length
val_ppl = math.exp(val_loss)
logger.info("[{}] PPL : {}".format("Output", val_ppl))