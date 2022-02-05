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
    def __init__(self, data = "yelp", min_freq=4, batch_size=512, device="cuda", embedding_size = 128, hidden_size = 500, style_size = 200, attn_size = 100, max_epoch = 18, seed = 0000, lr = 0.0005, eval_iter=500, max_iter=20000):
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

# Setting Config
config = StyleTransConfig()

# Set Seed
torch.manual_seed(config.seed)
np.random.seed(config.seed)

# Load Dataset
train, dev, test, train_iter, dev_iter, test_iter, X_VOCAB, C_LABEL = load_batch_iterator_with_eos(config.data_path, train="train.jsonl", val="dev.jsonl", test = "test.jsonl", batch_size=config.batch_size,device=config.device)

# Model Setting
input_size = len(X_VOCAB.vocab)
pad_idx = X_VOCAB.vocab.stoi["<pad>"]

# Criterion
nll = torch.nn.NLLLoss(ignore_index=X_VOCAB.vocab.stoi["<pad>"], reduction="mean")
BCE = torch.nn.CrossEntropyLoss()

# Load Models
enc_cls, attn_cls, senti_cls = get_classifier(input_size, config.embedding_size, config.attn_size , config.hidden_size , pad_idx, config.device, config.data, "cls")
enc_r, attn_r, senti_r= get_classifier(input_size, config.embedding_size, config.attn_size , config.hidden_size , pad_idx, config.device, config.data, "r")
enc_eval, attn_eval, senti_eval= get_classifier(input_size, config.embedding_size, config.attn_size , config.hidden_size , pad_idx, config.device, config.data, "eval")


ln0 = LayerNorm(200)
ln1 = LayerNorm(200)
enc = EncoderRNN(input_size, config.embedding_size, config.hidden_size, pad_idx, cls=False).to(config.device)
dec = DecoderRNN(vocab_size = len(X_VOCAB.vocab), max_len=None, hidden_size = config.hidden_size+config.style_size, sos_id=X_VOCAB.vocab.stoi["<bos>"], eos_id=X_VOCAB.vocab.stoi["<eos>"], pad_id=X_VOCAB.vocab.stoi["<pad>"], ln0 = ln0, ln1 = ln1, use_attention=False).to(config.device)

G = Generator(enc_cls, attn_cls, enc).to(config.device)
optim_G = torch.optim.Adam(list(G.parameters())+list(dec.parameters()), lr=0.0005)

for p in list(enc_r.parameters())+list(senti_r.parameters())+list(attn_r.parameters()):
    p.requires_grad = False

weights = (0.25, 0.25, 0.25, 0.25)

global_step = 0

name = "Ours_reproduce"

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
mse = torch.nn.MSELoss()
num_epoch = 400
enc_r.train()
attn_r.train()
senti_r.train()
enc_r.dropout.eval()
attn_r.dropout.eval()
senti_r.dropout.eval()
G.train()
dec.train()
gumbel = False
for epoch in range(num_epoch):
    kbar = pkbar.Kbar(target=math.ceil(len(train)/config.batch_size), width = 40)
    print("EPOCH [%d/%d] TRAINING" %(epoch+1, num_epoch))
    for i, batch in enumerate(train_iter):
        global_step+=1
        optim_G.zero_grad()


        batch_0 = my_batch(batch, 0)
        batch_1 = my_batch(batch, 1)

        batch_0_len = batch_0.X[1].max().cpu()
        batch_1_len = batch_1.X[1].max().cpu()

        hidden_2_0, output_2, attn_hidden, src_mask, _ = G(batch_0.X)
        decoder_outputs, decoder_hidden, ret_dict = dec(None, hidden_2_0.unsqueeze(0), output_2, style=0, mask = src_mask, transfer=False, attn_hidden = attn_hidden, label=batch_0.C.long(), gumbel = False, max_length = batch_0_len)
        
        
        
        
        hidden_2, output_2, attn_hidden, src_mask, reverse_score_0 = G(batch_0.X)
        decoder_outputs_1, decoder_hidden_1, ret_dict_1 = dec(None, hidden_2.unsqueeze(0), output_2, style=0, mask = src_mask, transfer=True, attn_hidden = attn_hidden, label=(1-batch_0.C).long(), gumbel=False,  max_length = batch_0_len)
        loss = 0
        batch_num = batch_0.C.size(0)
        self_loss = nll(torch.stack(decoder_outputs).squeeze().transpose(1, 2), batch_0.X[0])
        loss +=self_loss 
            
        hidden_2_1, output_2, attn_hidden, src_mask, _ = G(batch_1.X)
        decoder_outputs, decoder_hidden, ret_dict = dec(None, hidden_2_1.unsqueeze(0), output_2, style=1, mask = src_mask, transfer=False, attn_hidden = attn_hidden, label=batch_1.C.long(), gumbel = False, max_length = batch_1_len)
    
        
        
        
        
        hidden_2, output_2, attn_hidden, src_mask, reverse_score_1 = G(batch_1.X)
        decoder_outputs_0, decoder_hidden_0, ret_dict_0 = dec(None, hidden_2.unsqueeze(0), output_2, style=1, mask = src_mask, transfer=True, attn_hidden = attn_hidden, label=(1-batch_1.C).long(), gumbel=False, max_length = batch_1_len)
        batch_num = batch_1.C.size(0)
        self_loss = nll(torch.stack(decoder_outputs).squeeze().transpose(1, 2), batch_1.X[0])

        loss += self_loss
            
        loss.backward()
        s_loss = loss.item()
        #Get valid
        length = torch.Tensor(ret_dict_1["length"]).to(config.device)
        X_style_1 =(torch.stack(decoder_outputs_1).exp().squeeze(), length)
            

        length = torch.Tensor(ret_dict_0["length"]).to(config.device)
        X_style_0 =(torch.stack(decoder_outputs_0).exp().squeeze(), length)
        
        
        output, hidden = enc_r(X_style_1, soft=True)
        scores, attn_hidden, reverse_scores, src_mask = attn_r(output, X_style_1[1], temp=0.4)
        logits = senti_r(attn_hidden)
        loss_r = BCE(logits, torch.ones(logits.size(0), dtype=torch.long, device=config.device))
        
        
        
        
        output, hidden = enc_r(X_style_0, soft=True)
        scores, attn_hidden, reverse_scores, src_mask = attn_r(output, X_style_0[1], temp=0.4)
        logits = senti_r(attn_hidden)
        loss_r += BCE(logits, torch.zeros(logits.size(0), dtype=torch.long, device=config.device))
           

        cycle_loss = 0
        hidden, output, attn_hidden, src_mask,_ = G(X_style_1, soft=True)
        decoder_outputs, decoder_hidden, ret_dict = dec(None, hidden.unsqueeze(0), output, style=1, mask = src_mask, transfer=True, attn_hidden = attn_hidden, label=batch_0.C.long(), max_length = batch_0_len)
        
        mseLoss = mse(hidden, hidden_2_0.detach())
        
        
        
        batch_num = batch_0.C.size(0)
        cycle_loss += nll(torch.stack(decoder_outputs).squeeze().transpose(1, 2), batch_0.X[0])

        hidden, output, attn_hidden, src_mask,_ = G(X_style_0, soft=True)
        decoder_outputs, decoder_hidden, ret_dict = dec(None, hidden.unsqueeze(0), output, style=0, mask = src_mask, transfer=True, attn_hidden = attn_hidden, label=batch_1.C.long(), max_length = batch_1_len)
        mseLoss += mse(hidden, hidden_2_1.detach())
        
        batch_num = batch_1.C.size(0)
        cycle_loss += nll(torch.stack(decoder_outputs).squeeze().transpose(1, 2), batch_1.X[0])
            
        reward = loss_r.item()
        (0.2*loss_r+cycle_loss+mseLoss).backward()
        c_loss = cycle_loss.item()
        torch.nn.utils.clip_grad_norm_(list(G.parameters())+list(dec.parameters()), 30)
        optim_G.step()
        kbar.update(i, values=[("S_loss", s_loss), ("R_loss", reward), ("C_loss", c_loss)])
        with open("{}_log.txt".format(name), "a") as f:
            f.write("GLOBAL STEP {} - SelfLoss : {} CycleLoss : {} RewardLoss : {} \n".format(global_step, round(s_loss,3),round(c_loss,3), round(reward,3)))
        # if (global_step> 500 and global_step%50==0):
        if global_step%50==0:
            G.eval()
            dec.eval()
            result=[]
            original=[]
            st_label=[]
            acc=0
            total = 0
            perplexity = 0
            bleu_scores=0
            ref_bleu_scores = 0
            sentences=[]
            for i, (batch, batch_ref) in enumerate(zip(test_iter, test_iter_)):
                optim_G.zero_grad()
                batch_0 = my_batch(batch, 0)
                batch_1 = my_batch(batch, 1)
                batch_0_ref = my_batch(batch_ref, 0)
                batch_1_ref = my_batch(batch_ref, 1)


                loss = 0
                style_loss = 0
                if len(batch_0.C) !=0:
                    batch_0_len = batch_0.X[1].max().cpu()
                    hidden_2, output_2, attn_hidden, src_mask, _ = G(batch_0.X)
                    decoder_outputs_1, decoder_hidden_1, ret_dict_1 = dec(None, hidden_2.unsqueeze(0), output_2, style=0, mask = src_mask, transfer=True, attn_hidden = attn_hidden, label=(1-batch_0.C).long(), gumbel=False, max_length =batch_0_len)

                    length = torch.Tensor(ret_dict_1["length"]).to(config.device)
                    result.append(ret_dict_1["sequence"])
                    original.append(batch_0.X[0])
                    X_style_1 =(torch.stack(ret_dict_1["sequence"]).squeeze(), length)
                    text = torch.stack(ret_dict_1["sequence"]).squeeze()
                    srcmask = get_mask(length, text)
                    text = text.masked_fill(srcmask.T, X_VOCAB.vocab.stoi["<pad>"])

                    output, hidden = enc_cls(X_style_1, soft=False)
                    scores, attn_hidden, reverse_scores, src_mask = attn_cls(output, X_style_1[1])
                    logits = senti_cls(attn_hidden)
                    correct = (logits>0).sum().item()
                    total += logits.size(0)
                    acc += correct
                    style_loss += BCE(logits, torch.ones(logits.size(0), dtype=torch.long, device=config.device))
                    for t, leng, real, real_len in zip(text.T, length, batch_0.X[0].T, batch_0.X[1]):
                        pred = tensor_to_str(t[:int(leng.item()-1)],X_VOCAB).lower()
                        sentences.append(pred)
                        ref = [pred.split()]
                        candi = tensor_to_str(real[:int(real_len.item()-1)],X_VOCAB).lower().split()
                        score = sentence_bleu(ref, candi, weights, smoothing_function=SmoothingFunction().method1)
                        bleu_scores+=score
                    for t, leng, real, real_len in zip(text.T, length, batch_0_ref.X[0].T, batch_0_ref.X[1]):
                        ref = [tensor_to_str(t[:int(leng.item()-1)],X_VOCAB).lower().split()]
                        candi = tensor_to_str(real[:int(real_len.item()-1)],X_VOCAB).lower().split()
                        score = sentence_bleu(ref, candi, weights, smoothing_function=SmoothingFunction().method1)
                        ref_bleu_scores+=score
                if len(batch_1.C) !=0:
                    batch_1_len = batch_1.X[1].max().cpu()
                    hidden_2, output_2, attn_hidden, src_mask, _ = G(batch_1.X)
                    decoder_outputs_0, decoder_hidden_0, ret_dict_0 = dec(None, hidden_2.unsqueeze(0), output_2, style=1, mask = src_mask, transfer=True, attn_hidden = attn_hidden,  label=(1-batch_1.C).long(), gumbel=False, max_length = batch_1_len)
                    result.append(ret_dict_0["sequence"])
                    original.append(batch_1.X[0])
                    length = torch.Tensor(ret_dict_0["length"]).to(config.device)
                    X_style_0 =(torch.stack(ret_dict_0["sequence"]).squeeze(), length)
                    text = torch.stack(ret_dict_0["sequence"]).squeeze()
                    srcmask = get_mask(length, text)
                    text = text.masked_fill(srcmask.T, X_VOCAB.vocab.stoi["<pad>"])
                    output, hidden = enc_cls(X_style_0, soft=False)
                    scores, attn_hidden, reverse_scores, src_mask = attn_cls(output, X_style_0[1])

                    logits = senti_cls(attn_hidden)
                    correct = (logits<=0).sum().item()
                    style_loss += BCE(logits, torch.zeros(logits.size(0), dtype=torch.long, device=config.device))
                    total += logits.size(0)
                    acc += correct
                    for t, leng, real, real_len in zip(text.T, length, batch_1.X[0].T, batch_1.X[1]):
                        pred = tensor_to_str(t[:int(leng.item()-1)],X_VOCAB).lower()
                        sentences.append(pred)
                        ref = [pred.split()]
                        candi = tensor_to_str(real[:int(real_len.item()-1)],X_VOCAB).lower().split()
                        score = sentence_bleu(ref, candi, weights, smoothing_function=SmoothingFunction().method1)
                        bleu_scores+=score
                    for t, leng, real, real_len in zip(text.T, length, batch_1_ref.X[0].T, batch_1_ref.X[1]):
                        ref = [tensor_to_str(t[:int(leng.item()-1)],X_VOCAB).lower().split()]
                        candi = tensor_to_str(real[:int(real_len.item()-1)],X_VOCAB).lower().split()
                        score = sentence_bleu(ref, candi, weights, smoothing_function=SmoothingFunction().method1)
                        ref_bleu_scores+=score
            accuracy = round(acc/total, 3)
            ppl = evalPPL_yelp(model, sentences)
            bleu_scores = bleu_scores/1000*100
            ref_bleu_scores = ref_bleu_scores/1000*100
            with open("{}.txt".format(name), "a") as f:
                f.write("GLOBAL STEP {} Accuracy : {} \n".format(global_step, accuracy))
                f.write("GLOBAL STEP {} Perplexity : {} \n".format(global_step, ppl))
                f.write("GLOBAL STEP {} BLEU : {} \n".format(global_step, bleu_scores))
                f.write("GLOBAL STEP {} ref-BLEU : {} \n".format(global_step, ref_bleu_scores))
            if (accuracy > 0.86) and bleu_scores>55 and ppl<80:
                torch.save(G.state_dict(), "./model/{}_G_{}_{}_{}.pth".format(name, global_step, int(accuracy*100), int(ppl)))
                torch.save(dec.state_dict(), "./model/{}_dec_{}_{}_{}.pth".format(name,global_step, int(accuracy*100), int(ppl)))
            G.train()
            dec.train()
            enc_r.train()
            attn_r.train()
            senti_r.train()