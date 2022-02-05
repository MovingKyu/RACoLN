import pkbar
import math
from torch import nn, optim
import torch
import numpy as np
import torch.nn.functional as F
from src.util import *
from src.model import *
import logging
import os, sys
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

class StyleClsConfig:
    def __init__(self, data = "yelp", min_freq=4, batch_size=512, device="cuda", embedding_size = 128, hidden_size = 500, attn_size = 100, max_epoch = 18, seed = 0000, lr = 0.0005, eval_iter=500, max_iter=20000):
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
        self.eval_iter = eval_iter
        self.max_iter = max_iter

config = StyleClsConfig()
train, dev, test, train_iter, dev_iter, test_iter, X_VOCAB, C_LABEL = load_batch_iterator_with_eos(config.data_path, train="train.jsonl", val="dev.jsonl", test = "test.jsonl",
                        batch_size=config.batch_size,device=config.device)


embedding_dim = config.embedding_size
hidden_dim = config.hidden_size
input_size = len(X_VOCAB.vocab)
pad_idx = X_VOCAB.vocab.stoi["<pad>"]

# Train Three Classifier (1 for reverse attention, 1 for classification loss, and 1 for evaluation purpose)

enc_cls, attn_cls, senti_cls = get_classifier(input_size, config.embedding_size, config.attn_size , config.hidden_size , pad_idx, config.device, config.data, "cls")
enc_r, attn_r, senti_r= get_classifier(input_size, config.embedding_size, config.attn_size , config.hidden_size , pad_idx, config.device, config.data, "r")
enc_eval, attn_eval, senti_eval= get_classifier(input_size, config.embedding_size, config.attn_size , config.hidden_size , pad_idx, config.device, config.data, "eval")

models = [enc_r, attn_r, senti_r, enc_cls, attn_cls, senti_cls, enc_eval, attn_eval, senti_eval]

max_acc1 = 0
max_acc2 = 0
max_acc3 = 0

for model in models:
    model.eval()
acc1 = 0
acc2 = 0
acc3 = 0
for i, batch in enumerate(test_iter):
    #reward
    logits = runClassifier(batch, enc_r, attn_r, senti_r)
    acc1 += computeAccuracy(logits, batch.C)
    
    #cls
    logits = runClassifier(batch, enc_cls, attn_cls, senti_cls)
    acc2 += computeAccuracy(logits, batch.C)
    
    #eval
    logits = runClassifier(batch, enc_eval, attn_eval, senti_eval)
    acc3 += computeAccuracy(logits, batch.C)

logger.info("Acc - reward - {}".format(acc1/len(test)))
logger.info("Acc - classifier - {}".format(acc2/len(test)))
logger.info("Acc - evaluator - {}".format(acc3/len(test)))