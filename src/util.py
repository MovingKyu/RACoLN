from torchtext.legacy.data import TabularDataset, BucketIterator, Iterator,Field, LabelField
import torch
from src.model import *
import math
import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

bleu_weight = (0.25, 0.25, 0.25, 0.25)

def load_batch_iterator_with_eos(path, train, val, test, batch_size, device):
    split_tokenizer = lambda x: x.split()
    X = Field(tokenize=split_tokenizer, lower=True, include_lengths=True, eos_token="<eos>")
    C = LabelField(dtype=torch.long)
    train, dev, test = TabularDataset.splits(path=path, train=train,
                            validation=val, test=test,
                      format="json",
                      fields={"X":("X", X),
                              "C": ('C', C)})

    train_iter = BucketIterator(train, batch_size = batch_size, sort_key=lambda x: len(x.X), device=torch.device(device), shuffle=True)
    dev_iter = BucketIterator(dev, batch_size = batch_size, sort_key=None, device=torch.device(device), sort=None, sort_within_batch=None)
    test_iter = Iterator(test, batch_size = batch_size, sort_key=None, device=torch.device(device), sort=False, sort_within_batch=False,train=False)

    X.build_vocab(train, specials=["<bos>"],min_freq=4)
    C.build_vocab(train)
    return train, dev, test, train_iter, dev_iter, test_iter, X, C

#Create jsonl file for torchtext format
def create_json(data, phase):
    import json
    with open("./Data/{}/sentiment.{}.0".format(data, phase), "r") as f:
        text_0 = f.readlines()
    with open("./Data/{}/sentiment.{}.1".format(data, phase), "r") as f:
        text_1 = f.readlines()
    with open('./Data/{}/sentiment.{}.jsonl'.format(data, phase), 'w') as fp:
        for i, line in enumerate(text_0):
            data_json={}
            data_json["index"] = i
            data_json["X"] = line.replace("\n","")
            data_json["Y"] = line.replace("\n","")
            data_json["C"] = 0
            json.dump(data_json, fp)
            fp.write("\n")
        for i, line in enumerate(text_1):
            data_json={}
            data_json["index"] = i
            data_json["X"] = line.replace("\n","")
            data_json["Y"] = line.replace("\n","")
            data_json["C"] = 1
            json.dump(data_json, fp)
            fp.write("\n")

def find_index(word, Vocab):
    return Vocab.vocab.stoi[word]
def find_word(index, Vocab):
    return Vocab.vocab.itos[index]

#Tensor to string to check
def tensor_to_str(tensor, Vocab):
    return " ".join([find_word(word, Vocab) for word in tensor.tolist()])

def runClassifier(batch, enc, attn, cls):
    # RETURN logits
    output, _= enc(batch.X)
    _, attn_hidden, _, _= attn(output, batch.X[1])
    logits = cls(attn_hidden)
    return logits

def computeAccuracy(logits, label):
    _, pred = logits.max(dim=-1)
    return (pred==label).sum().item()
    
def saveModel(name, model):
    torch.save(model.state_dict(), f"model/{name}.pth")

def get_length(tokens, eos_idx):
    tokens = tokens.squeeze()
    lengths = torch.cumsum(tokens == eos_idx, 0)
    lengths = (lengths==0).long().sum(0)
    lengths = lengths+1
    index = lengths>tokens.size(0)
    lengths[index] = tokens.size(0)
    return lengths.detach()

def get_mask(length, text):
    max_len = text.size(0)
    #max_len = max(length).item()
    pos_idx = torch.arange(max_len)
    pos_inx = pos_idx.expand(length.size(0), int(max_len))
    pos_idx = pos_idx.to(length.device)
    src_mask_nll = pos_idx>=(length).unsqueeze(-1)
    return src_mask_nll

def get_classifier(input_size, embedding_dim, attention_dim, hidden_dim, pad_idx, device, data, model_name):
    enc = EncoderRNN(input_size, embedding_dim, hidden_dim, pad_idx).to(device)
    enc.load_state_dict(torch.load(f"model/{data}_enc_{model_name}.pth"))
    attn = MLPAttention(hidden_dim, attention_dim).to(device)
    attn.load_state_dict(torch.load(f"model/{data}_attn_{model_name}.pth"))
    senti = SentimentClassifier(hidden_dim).to(device)
    senti.load_state_dict(torch.load(f"model/{data}_senti_{model_name}.pth"))
    return enc, attn, senti

def prepareBatchForLM(text, bos_idx, pad_idx, eos_idx):
    text[text==eos_idx] = pad_idx
    bos = torch.full((1, text.size(1)),bos_idx).to(text.device)
    text = text[:-1,:]
    text = torch.cat([bos, text], dim=0)
    return text

def processOutput(dictionary, pad_idx, device):
    length = torch.Tensor(dictionary["length"]).to(device)
    soft_text =(torch.stack(dictionary["sequence"]).squeeze(), length)
    text = torch.stack(dictionary["sequence"]).squeeze()
    srcmask = get_mask(length, text)
    text = text.masked_fill(srcmask.T, pad_idx)
    return length, soft_text, text, srcmask

def computeSelfLoss(batch, G, dec, nll, reduce=True):
    max_len = batch.X[1].max().cpu()

    # Self Reconstruction
    hidden_self, output_selt, attn_hidden_self, src_mask, _ = G(batch.X)
    decoder_outputs, decoder_hidden, ret_dict = dec(None, hidden_self.unsqueeze(0), output_selt, mask = src_mask, attn_hidden = attn_hidden_self, label=batch.C, gumbel = False, max_length = max_len)
    
    self_loss = nll(torch.stack(decoder_outputs).squeeze().transpose(1, 2), batch.X[0])
    if reduce:
        self_loss = self_loss.sum()/batch.X[1].sum()
    return self_loss, hidden_self
def computeOtherLoss(batch, G, dec, enc_r, attn_r, senti_r, hidden_self, BCE, mse, nll, device, reduce=True):
    max_len = batch.X[1].max().cpu()
    # Style Transfer 
    hidden_tsf, output_tsf, attn_hidden_tsf, src_mask, _ = G(batch.X)
    decoder_outputs_tsf, _ , output_dictionary_tsf = dec(None, hidden_tsf.unsqueeze(0), output_tsf, mask = src_mask, attn_hidden = attn_hidden_tsf, label=(1-batch.C), gumbel=False,  max_length = max_len)

    #Get valid
    length = torch.Tensor(output_dictionary_tsf["length"]).to(device)
    X_style =(torch.stack(decoder_outputs_tsf).exp().squeeze(), length)
        
    output, hidden = enc_r(X_style)
    scores, attn_hidden, reverse_scores, src_mask = attn_r(output, X_style[1], temp=1.0)
    logits = senti_r(attn_hidden)
    loss_r = BCE(logits, 1-batch.C)

    hidden, output, attn_hidden, src_mask,_ = G(X_style)
    decoder_outputs, decoder_hidden, ret_dict = dec(None, hidden.unsqueeze(0), output, mask = src_mask, attn_hidden = attn_hidden, label=batch.C, max_length = max_len)
    
    mseLoss = mse(hidden, hidden_self.detach())
    
    cycle_loss = nll(torch.stack(decoder_outputs).squeeze().transpose(1, 2), batch.X[0]) 
    if reduce:
        cycle_loss = cycle_loss.sum()/batch.X[1].sum()
        loss_r = loss_r.sum()/batch.C.size(0)
        mseLoss = mseLoss.sum()/(batch.C.size(0)*mseLoss.size(1))
    return loss_r, mseLoss, cycle_loss

def getLossAndMetrics(iter,G, dec, nll, enc_r, attn_r, senti_r, BCE, mse, config, pad_idx, enc_cls, attn_cls, senti_cls, iter_len, X_VOCAB, C_LABEL, save=False):
    acc, total, bleu_scores = 0, 0, 0
    sentences=[]
    labels = []
    self_loss_list, style_loss_list, mse_loss_list, cycle_loss_list = 0,0,0,0
    for i, batch in enumerate(iter):
        # Compute Loss
        self_loss, hidden_self = computeSelfLoss(batch, G, dec, nll)
        loss_r, mseLoss, cycle_loss = computeOtherLoss(batch, G, dec, enc_r, attn_r, senti_r, hidden_self, BCE, mse, nll, config.device)
        self_loss_list += self_loss.item() * batch.C.size(0)
        style_loss_list+= loss_r.item()* batch.C.size(0)
        mse_loss_list+= mseLoss.item()* batch.C.size(0)
        cycle_loss_list+= cycle_loss.item()* batch.C.size(0)


        max_len = batch.X[1].max().cpu()
        hidden, output, attn_hidden, src_mask, _ = G(batch.X)
        _, _, output_dictionary = dec(None, hidden.unsqueeze(0), output, mask = src_mask, attn_hidden = attn_hidden, label=(1-batch.C), gumbel=False, max_length =max_len)
        length, soft_text, text, srcmask = processOutput(output_dictionary, pad_idx, config.device)

        output, hidden = enc_cls(soft_text)
        scores, attn_hidden, reverse_scores, src_mask = attn_cls(output, soft_text[1])
        logits = senti_cls(attn_hidden)
        _, preds = logits.max(dim=-1)
        correct = (preds==(1-batch.C)).sum().item()
        total += logits.size(0)
        acc += correct
        labels.extend((1-batch.C).tolist())
        for t, leng, real, real_len in zip(text.T, length, batch.X[0].T, batch.X[1]):
            pred = tensor_to_str(t[:int(leng.item()-1)],X_VOCAB).lower()
            sentences.append(pred)
            ref = [pred.split()]
            candi = tensor_to_str(real[:int(real_len.item()-1)],X_VOCAB).lower().split()
            score = sentence_bleu(ref, candi, bleu_weight, smoothing_function=SmoothingFunction().method1)
            bleu_scores+=score
    accuracy = round(acc/total, 3)
    # ppl = evalPPL_yelp(model, sentences)
    bleu_scores = bleu_scores/iter_len*100
    loss = self_loss_list/iter_len + config.styleLossCoef*style_loss_list/iter_len + mse_loss_list/iter_len + cycle_loss_list/iter_len
    if save:
        with open('output/{}.jsonl'.format(config.name), 'w') as fp:
            for i, line in enumerate(zip(sentences, labels)):
                data_json={}
                data_json["index"] = i
                data_json["X"] = line[0].replace("\n","")
                data_json["C"] = C_LABEL.vocab.itos[line[1]]
                json.dump(data_json, fp)
                fp.write("\n")
    return loss, accuracy, bleu_scores

def getLogger(LoggerName):
    import logging
    import os
    import sys
    logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
    )
    logger = logging.getLogger(LoggerName)
    return logger

def computePPL(iter, lm, criterion, bos_idx, pad_idx, eos_idx):
    ppl = []
    length = 0
    for i, batch in enumerate(iter):
        lm_text = prepareBatchForLM(batch.X[0], bos_idx, pad_idx, eos_idx)
        logit = lm(lm_text.T, batch.X[1])
        logit = logit.view(-1, logit.size(2))
        target = torch.reshape(batch.X[0].T, (-1,))
        ppl.extend(criterion(logit, target).tolist())
        length += batch.X[1].sum()
    loss = sum(ppl)/length
    ppl = math.exp(loss)
    return ppl 

def fetchIter(path, X_VOCAB, C_LABEL, batch_size, device):
    dataset= TabularDataset(path=path,
                    format="json",
                    fields={"X":("X", X_VOCAB),
                            "C": ('C', C_LABEL)})
    iter = BucketIterator(dataset, batch_size=batch_size,sort_key=lambda x: len(x.X), device=device, shuffle=False)
    return iter

def getSpecialTokens(X_VOCAB):
    pad_idx = X_VOCAB.vocab.stoi["<pad>"]
    bos_idx = X_VOCAB.vocab.stoi["<bos>"]
    eos_idx = X_VOCAB.vocab.stoi["<eos>"]
    return bos_idx, pad_idx, eos_idx
    