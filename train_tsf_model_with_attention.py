import pkbar
import math
import torch
from src.model import *
from src.util import *
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from torchtext.legacy.data import TabularDataset, BucketIterator
import kenlm
import math
import numpy as np

class StyleTransConfig:
    def __init__(self, data = "yelp", min_freq=4, batch_size=512, device="cuda", embedding_size = 128, hidden_size = 500, style_size = 200, attn_size = 100, max_epoch = 40, seed = 0000, lr = 0.0005, eval_iter=100, max_iter=20000, num_class=2, styleLossCoef=0.15, useAttention=True):
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
        self.eval_iter = eval_iter
        self.max_iter = max_iter
        self.style_size = style_size 
        self.num_class = num_class
        self.styleLossCoef= styleLossCoef
        self.useAttention = useAttention

# Setting Config
config = StyleTransConfig()

# Logger
logger = getLogger("train_tsf_model.py")


# Set Seed
torch.manual_seed(config.seed)
np.random.seed(config.seed)

# Load Dataset
train, dev, test, train_iter, dev_iter, test_iter, X_VOCAB, C_LABEL = load_batch_iterator_with_eos(config.data_path, train="train.jsonl", val="dev.jsonl", test = "test.jsonl", batch_size=config.batch_size,device=config.device)

# Model Setting
input_size = len(X_VOCAB.vocab)
pad_idx = X_VOCAB.vocab.stoi["<pad>"]


# Load Models
enc_cls, attn_cls, senti_cls = get_classifier(input_size, config.embedding_size, config.attn_size , config.hidden_size , pad_idx, config.device, config.data, "cls")
enc_r, attn_r, senti_r= get_classifier(input_size, config.embedding_size, config.attn_size , config.hidden_size , pad_idx, config.device, config.data, "r")
enc_eval, attn_eval, senti_eval= get_classifier(input_size, config.embedding_size, config.attn_size , config.hidden_size , pad_idx, config.device, config.data, "eval")


enc = EncoderRNN(input_size, config.embedding_size, config.hidden_size, pad_idx).to(config.device)
dec = DecoderRNN_LN(vocab_size = len(X_VOCAB.vocab), max_len=None, hidden_size = config.hidden_size+config.style_size, sos_id=X_VOCAB.vocab.stoi["<bos>"], eos_id=X_VOCAB.vocab.stoi["<eos>"], pad_id=X_VOCAB.vocab.stoi["<pad>"], style_size=config.style_size, num_class=config.num_class, use_attention=True).to(config.device)

for p in list(enc_r.parameters())+list(senti_r.parameters())+list(attn_r.parameters()):
    p.requires_grad = False

G = Generator(enc_cls, attn_cls, enc).to(config.device)
optim_G = torch.optim.Adam(list(G.parameters())+list(dec.parameters()), lr=0.0005)


weights = (0.25, 0.25, 0.25, 0.25)

global_step = 0
name = "{}-{}".format(config.data, config.styleLossCoef)
if config.useAttention:
    name = "{}-{}-withAttention".format(config.data, config.styleLossCoef)

config.name = name

split_tokenizer = lambda x: x.split()
test_ = TabularDataset(path="./data/yelp/sentiment.ref.jsonl",
                  format="json",
                  fields={"X":("X", X_VOCAB),
                          "C": ('C', C_LABEL)})

test_iter_ = BucketIterator(test_, batch_size=config.batch_size,sort_key=lambda x: len(x.X), device=config.device, shuffle=False)
test_acc_sum = 0.0
model = kenlm.LanguageModel('yelp.binary')


G.train()
dec.train()
# Criterion
nll = torch.nn.NLLLoss(ignore_index=X_VOCAB.vocab.stoi["<pad>"], reduction="none")
BCE = torch.nn.CrossEntropyLoss(reduction="none")
mse = torch.nn.MSELoss(reduction="none")

G.train()
dec.train()
gumbel = False
minLoss = float("inf")
for epoch in range(config.max_epoch):
    kbar = pkbar.Kbar(target=math.ceil(len(train)/config.batch_size), width = 40)
    print("EPOCH [%d/%d] TRAINING" %(epoch+1, config.max_epoch))
    for i, batch in enumerate(train_iter):
        global_step+=1
        optim_G.zero_grad()

        self_loss, hidden_self = computeSelfLoss(batch, G, dec, nll)
        self_loss.backward()
        s_loss = self_loss.item()
        
        loss_r, mseLoss, cycle_loss = computeOtherLoss(batch, G, dec, enc_r, attn_r, senti_r, hidden_self, BCE, mse, nll, config.device)
        reward = loss_r.item()
        c_loss = cycle_loss.item()
        (config.styleLossCoef*loss_r+cycle_loss+mseLoss).backward()

        # Clip Grad
        torch.nn.utils.clip_grad_norm_(list(G.parameters())+list(dec.parameters()), 30)
        optim_G.step()
        kbar.update(i, values=[("S_loss", s_loss), ("R_loss", reward), ("C_loss", c_loss)])
        with open("logs/{}_log.txt".format(config.name), "a") as f:
            f.write("GLOBAL STEP {} - SelfLoss : {} CycleLoss : {} RewardLoss : {} \n".format(global_step, round(s_loss,3),round(c_loss,3), round(reward,3)))
        if global_step%config.eval_iter==0:
            G.eval()
            dec.eval()
            loss, accuracy, ppl, bleu_scores= getLossAndMetrics(dev_iter,G, dec, nll, enc_r, attn_r, senti_r, BCE, mse, config, pad_idx, enc_cls, attn_cls, senti_cls, model, weights, len(dev), X_VOCAB)
            with open("logs/{}.txt".format(config.name), "a") as f:
                f.write("GLOBAL STEP {} Loss : {} \n".format(global_step, loss))
                f.write("GLOBAL STEP {} Accuracy : {} \n".format(global_step, accuracy))
                f.write("GLOBAL STEP {} Perplexity : {} \n".format(global_step, ppl))
                f.write("GLOBAL STEP {} BLEU : {} \n".format(global_step, bleu_scores))
                if loss<minLoss:
                    minLoss = loss
                    torch.save(G.state_dict(), "./model/{}_G.pth".format(config.name))
                    torch.save(dec.state_dict(), "./model/{}_dec.pth".format(config.name))
                    f.write("GLOBAL STEP {} Saving Checkpoint with Loss {} \n".format(global_step, loss))
                    loss, accuracy, ppl, bleu_scores= getLossAndMetrics(test_iter,G, dec, nll, enc_r, attn_r, senti_r, BCE, mse, config, pad_idx, enc_cls, attn_cls, senti_cls, model, weights, len(test), X_VOCAB, save=True)
                    f.write("[TEST] GLOBAL STEP {} Loss : {} \n".format(global_step, loss))
                    f.write("[TEST] GLOBAL STEP {} Accuracy : {} \n".format(global_step, accuracy))
                    f.write("[TEST] GLOBAL STEP {} Perplexity : {} \n".format(global_step, ppl))
                    f.write("[TEST] GLOBAL STEP {} BLEU : {} \n".format(global_step, bleu_scores))
            G.train()
            dec.train()