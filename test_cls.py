import torch
from src.util import *
from src.model import *
import logging
import os, sys
import argparse
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("test_cls.py")

parser = argparse.ArgumentParser(description='Argparse For Classifiers')
parser.add_argument('--embedding-size', type=int, default=128,
                    help='yelp set to 128')
parser.add_argument('--hidden-size', type=int, default=500,
                    help='hidden size set to 500')
parser.add_argument('--batch-size', type=int, default=512,
                    help='batch size set to 512 for yelp')
parser.add_argument('--attn-size', type=int, default=100,
                    help='attn size set to 100 for yelp')
parser.add_argument('--path-to-output', type=str, default="output/yelp_racoln.jsonl",
                    help='jsonl path')
parser.add_argument('--data', type=str, default="yelp",
                    help='data')
config = parser.parse_args()

if torch.cuda.is_available():
    config.device = "cuda"
else:
    config.device = "cpu"
config.data_path = f"data/{config.data}"

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