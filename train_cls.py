import pkbar
import math
import torch
import argparse
from src import util
from src.model import *
import numpy as np
import random

parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--embedding-size', type=int, default=128,
                    help='yelp set to 128')
parser.add_argument('--hidden-size', type=int, default=500,
                    help='hidden size set to 500')
parser.add_argument('--batch-size', type=int, default=512,
                    help='batch size set to 512 for yelp')
parser.add_argument('--attn-size', type=int, default=100,
                    help='attn size set to 512 for yelp')
parser.add_argument('--data', type=str, default="yelp",
                    help='data')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--seed', type=int, default=0000,
                    help='random seed')
parser.add_argument('--max-epoch', type=int, default=20,
                    help='max epoch')
parser.add_argument('--log_iter', type=int, default=100,
                    help='log iteration')
parser.add_argument('--eval_iter', type=int, default=500,
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

train, dev, test, train_iter, dev_iter, test_iter, X_VOCAB, C_LABEL = util.load_batch_iterator_with_eos(config.data_path, train="train.jsonl", val="dev.jsonl", test = "test.jsonl",
                        batch_size=config.batch_size,device=config.device)


embedding_dim = config.embedding_size
hidden_dim = config.hidden_size
input_size = len(X_VOCAB.vocab)
pad_idx = X_VOCAB.vocab.stoi["<pad>"]

# Train Three Classifier (1 for reverse attention, 1 for classification loss, and 1 for evaluation purpose)

enc_cls = EncoderRNN(input_size, embedding_dim, hidden_dim, pad_idx).to(config.device)
attn_cls = MLPAttention(hidden_dim, config.attn_size).to(config.device)
senti_cls = SentimentClassifier(hidden_dim).to(config.device)
cls_optim = torch.optim.Adam(list(enc_cls.parameters())+list(attn_cls.parameters())+list(senti_cls.parameters()))

enc_r = EncoderRNN(input_size, embedding_dim, hidden_dim, pad_idx).to(config.device)
attn_r = MLPAttention(hidden_dim, config.attn_size).to(config.device)
senti_r = SentimentClassifier(hidden_dim).to(config.device)
r_optim = torch.optim.Adam(list(attn_r.parameters())+list(enc_r.parameters())+list(senti_r.parameters()))

enc_eval = EncoderRNN(input_size, embedding_dim, hidden_dim, pad_idx).to(config.device)
attn_eval = MLPAttention(hidden_dim, config.attn_size).to(config.device)
senti_eval = SentimentClassifier(hidden_dim).to(config.device)
eval_optim = torch.optim.Adam(list(enc_eval.parameters())+list(attn_eval.parameters())+list(senti_eval.parameters()))

models = [enc_r, attn_r, senti_r, enc_cls, attn_cls, senti_cls, enc_eval, attn_eval, senti_eval]

ce = torch.nn.CrossEntropyLoss()

num_epoch = config.max_epoch
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
        if i%config.eval_iter==0 and epoch != 0:
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