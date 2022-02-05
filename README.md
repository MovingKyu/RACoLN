# RACoLN Official Implementation

This repository is the official pytorch implementation of the [paper](https://aclanthology.org/2021.acl-long.8/) "Enhancing Content Preservation in Text Style Transfer Using Reverse Attention and Conditional Layer Normalization" which was presented at ACL 2021 main conference as a long paper.

# Tested Environment
- pytorch == 1.9.0
- pytyhon == 3.7.11
- nltk == 3.6.7
- torchtext == 0.10.0
- pkbar == 0.5

# Dataset
Our implementation used torchtext, hence we have changed the input format to jsonl. If you want to try the model on other dataset, please change the input format accordingly (you can check at data directory)

# Training classifiers
In this work, we train three classifiers: 1) for reverse attention, 2) for style loss, and 3) for evaluation purpose. The classifier is made of GRU and Attention network.

The configuration is defined as the default option in the file. If you would like to try a different value, check the argparse options. 

```
python train_cls.py
```
# Testing the classifiers on the test set

In order to test the trained classifiers, you run the following:
```
python test_cls.py
```
The accuracy should be between 97.5 and 98.0 for Yelp dataset.

# Training language model
Different from the original paper, where we have used KenLM, this repository trains a GRU-based langauge model as we can skip installing kenLM. (Although we use GRU-based LM, we have checked that the output will have similar PPL score with KenLM).

To train the langauge model for evaluation purpose, computing Perplexity, run the following:

```
python train_lm.py
```

# Testing the language model
```
python test_lm.py
```
The code will output the PPL score on test set, which should be around 33.

# Training Transfer Model (RACoLN)
```
python train_tsf_model.py
```

The code will start trainining the main model of the paper.

One minor change is made on the balancing parameter. In the original paper, we have normalized the total loss with number of sentences in a batch. In order to handle variable length of a corpus, this repository now normalizes the total loss with the number of tokens in a batch.

The result should be similar to the ones reported in the paper. With minor change in the balancin parameter, the PPL and ref-BLEU are slightly better while self-BLEU is slightly decreased.

|                 | Style Acc | Self-BLEU | Ref-BLEU | PPL   |
|-----------------|-----------|-----------|----------|-------|
| RACoLN | 90.9      | 58.73     | 20.67    | 47.18 |

# Reference
```
@inproceedings{lee-etal-2021-enhancing,
    title = "Enhancing Content Preservation in Text Style Transfer Using Reverse Attention and Conditional Layer Normalization",
    author = "Lee, Dongkyu  and
      Tian, Zhiliang  and
      Xue, Lanqing  and
      Zhang, Nevin L.",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.8",
    doi = "10.18653/v1/2021.acl-long.8",
    pages = "93--102",
}
```

