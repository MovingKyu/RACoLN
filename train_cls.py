import pkbar
import math
from torch import nn, optim
import torch
import numpy as np
import torch.nn.functional as F


from src import util
from src.model import *

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
train, dev, test, train_iter, dev_iter, test_iter, X_VOCAB, C_LABEL = util.load_batch_iterator_with_eos(config.data_path, train="train.jsonl", val="dev.jsonl", test = "test.jsonl",
                        batch_size=config.batch_size,device=config.device)


embedding_dim = config.embedding_size
hidden_dim = config.hidden_size
input_size = len(X_VOCAB.vocab)
pad_idx = X_VOCAB.vocab.stoi["<pad>"]

# Train Three Classifier (1 for reverse attention, 1 for classification loss, and 1 for evaluation purpose)

enc_cls = EncoderRNN(input_size, embedding_dim, hidden_dim, pad_idx, cls=True).to(config.device)
attn_cls = MLPAttention(hidden_dim, 100).to(config.device)
senti_cls = SentimentClassifier(hidden_dim).to(config.device)
cls_optim = torch.optim.Adam(list(enc_cls.parameters())+list(attn_cls.parameters())+list(senti_cls.parameters()))

enc_r = EncoderRNN(input_size, embedding_dim, hidden_dim, pad_idx, cls=True).to(config.device)
attn_r = MLPAttention(hidden_dim, 100).to(config.device)
senti_r = SentimentClassifier(hidden_dim).to(config.device)
r_optim = torch.optim.Adam(list(attn_r.parameters())+list(enc_r.parameters())+list(senti_r.parameters()))

enc_eval = EncoderRNN(input_size, embedding_dim, hidden_dim, pad_idx, cls=True).to(config.device)
attn_eval = MLPAttention(hidden_dim, 100).to(config.device)
senti_eval = SentimentClassifier(hidden_dim).to(config.device)
eval_optim = torch.optim.Adam(list(enc_eval.parameters())+list(attn_eval.parameters())+list(senti_eval.parameters()))

models = [enc_r, attn_r, senti_r, enc_cls, attn_cls, senti_cls, enc_eval, attn_eval, senti_eval]

ce = torch.nn.CrossEntropyLoss()

num_epoch = 10
enc_cls.train()
attn_cls.train()
senti_cls.train()
enc_r.train()
attn_r.train()
senti_r.train()
max_acc1 = 0
max_acc2 = 0
max_acc3 = 0

for epoch in range(num_epoch):
    kbar = pkbar.Kbar(target=math.ceil(len(train)/config.batch_size), width = 40)
    print("EPOCH [%d/%d] TRAINING" %(epoch+1, num_epoch))
    for i, batch in enumerate(train_iter):
        r_optim.zero_grad()
        cls_optim.zero_grad()
        eval_optim.zero_grad()
        
        #reward
        logits = util.runClassifier(batch, enc_r, attn_r, senti_r)
        loss_r = ce(logits, batch.C)
        loss_r.backward()
        
        #cls
        logits = util.runClassifier(batch, enc_cls, attn_cls, senti_cls)
        loss_cls = ce(logits, batch.C)
        loss_cls.backward()
        
        #eval
        logits = util.runClassifier(batch, enc_eval, attn_eval, senti_eval)
        loss_eval= ce(logits, batch.C)
        loss_eval.backward()

        # Update Paramter
        cls_optim.step()
        r_optim.step()
        eval_optim.step()

        kbar.update(i, values=[("loss(R)", loss_r.item()), ("loss(C)", loss_cls.item()), ("loss(E)", loss_eval.item())])
        if i%config.eval_iter==0:
            for model in models:
                model.eval()
            acc1 = 0
            acc2 = 0
            acc3 = 0
            for i, batch in enumerate(dev_iter):
                #reward
                logits = util.runClassifier(batch, enc_r, attn_r, senti_r)
                acc1 += util.computeAccuracy(logits, batch.C)
                
                #cls
                logits = util.runClassifier(batch, enc_cls, attn_cls, senti_cls)
                acc2 += util.computeAccuracy(logits, batch.C)
                
                #eval
                logits = util.runClassifier(batch, enc_eval, attn_eval, senti_eval)
                acc3 += util.computeAccuracy(logits, batch.C)

                # len(dev)
            if acc1>max_acc1:
                max_acc1 = acc1
                util.saveModel(f'{config.data}_enc_r', enc_r)
                util.saveModel(f'{config.data}_attn_r', attn_r)
                util.saveModel(f'{config.data}_senti_r', senti_r)
            if acc2>max_acc2:
                max_acc2 = acc2
                util.saveModel(f'{config.data}_enc_cls', enc_cls)
                util.saveModel(f'{config.data}_attn_cls', attn_cls)
                util.saveModel(f'{config.data}_senti_cls', senti_cls)
            if acc3>max_acc3:
                max_acc3 = acc3
                util.saveModel(f'{config.data}_enc_eval', enc_eval)
                util.saveModel(f'{config.data}_attn_eval', attn_eval)
                util.saveModel(f'{config.data}_senti_eval', senti_eval)

            for model in models:
                model.train()
    print()