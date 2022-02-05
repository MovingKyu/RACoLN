import pkbar
import math
from torch import nn, optim
import torch
import numpy as np
import torch.nn.functional as F
import math
from src.util import *
from src.model import *
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

config = LmConfig()
train, dev, test, train_iter, dev_iter, test_iter, X_VOCAB, C_LABEL = load_batch_iterator_with_eos(config.data_path, train="train.jsonl", val="dev.jsonl", test = "test.jsonl",
                        batch_size=config.batch_size,device=config.device)

bos_idx, pad_idx, eos_idx = getSpecialTokens(X_VOCAB)

# Fetch LM
lm = GRU_LM(len(X_VOCAB.vocab), config.embedding_size, config.hidden_size, pad_idx, dropout = 0.4).to(config.device)
lm.load_state_dict(torch.load("model/{}_lm.pth".format(config.data)))

# Fetch Classifier
enc_eval, attn_eval, senti_eval= get_classifier(len(X_VOCAB.vocab), config.embedding_size, config.attn_size , config.hidden_size , pad_idx, config.device, config.data, "eval")


# Fetch Iter
output_iter = fetchIter("output/yelp-0.15_.jsonl", X_VOCAB, C_LABEL, config.batch_size, config.device)

ref_iter = fetchIter(f"data/{config.data}/sentiment.ref.jsonl",X_VOCAB, C_LABEL, config.batch_size, config.device)
test_iter = fetchIter(f"data/{config.data}/test.jsonl",X_VOCAB, C_LABEL, config.batch_size, config.device)

ce_val = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_idx)
lm.eval()
enc_eval.eval()
attn_eval.eval()
senti_eval.eval()

# Style Accuracy Computation
numCorrect = 0
for batch in output_iter:
    logits = runClassifier(batch, enc_eval, attn_eval, senti_eval)
    numCorrect += computeAccuracy(logits, batch.C)
logger.info("[{}] StyleACC : {}".format("Output", numCorrect/len(test)*100))

# BLEU Computation
selfBLEU = 0
refBLEU = 0
for batch_output, batch_ref, batch_test in zip(output_iter, ref_iter, test_iter):
    for text, text_length, test_text, test_len, ref_text, ref_len in zip(batch_output.X[0].T, batch_output.X[1], batch_test.X[0].T, batch_test.X[1], batch_ref.X[0].T, batch_ref.X[1]):
        pred = tensor_to_str(text[:int(text_length.item()-1)],X_VOCAB).lower()
        ref = [pred.split()]
        candi = tensor_to_str(test_text[:int(test_len.item()-1)],X_VOCAB).lower().split()
        score = sentence_bleu(ref, candi, (0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
        selfBLEU+=score
        candi = tensor_to_str(ref_text[:int(ref_len.item()-1)],X_VOCAB).lower().split()
        score = sentence_bleu(ref, candi, (0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
        refBLEU+=score

logger.info("[{}] Self-BLEU : {}".format("Output", selfBLEU/len(test)*100))
logger.info("[{}] Ref-BLEU : {}".format("Output", refBLEU/len(test)*100))

# Perplexity Computation
ppl = computePPL(output_iter, lm, ce_val, bos_idx, pad_idx, eos_idx)
logger.info("[{}] PPL : {}".format("Output", ppl))