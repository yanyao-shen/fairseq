# Introduction
This is an implementation of the DenseNMT architecture described in the paper 'Dense Information Flow for Neural Machine Translation':
```
@inproceedings{shen2018dense,
  title={Dense Information Flow for Neural Machine Translation},
  author={Shen, Yanyao and Tan, Xu and He, Di and Qin, Tao and Liu, Tie-Yan},
  booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
  volume={1},
  pages={1294--1303},
  year={2018}
}
```
It is built based on fairseq, a sequence-to-sequence learning toolkit for [Torch](http://torch.ch/) from Facebook AI Research tailored to Neural Machine Translation (NMT).

![Model](fairseq.gif)

# Requirements and Installation
* The related packages are listed in the github page of the [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq) project. In short, we need: gpu, nccl, torch, nn. For more details for installation, please check [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq) .
* For data preprocessing, you may need other packages, such as [subword-nmt](https://github.com/rsennrich/subword-nmt) for BPE level training.   

Install fairseq by cloning the GitHub repository and running
```
luarocks make rocks/fairseq-scm-1.rockspec
```

# Training a New Model

## Data Pre-processing
The fairseq source distribution contains an example pre-processing script for
the IWSLT14 German-English corpus.
Pre-process and binarize the data as follows:
```
$ cd data/
$ bash prepare-iwslt14.sh
$ cd ..
$ TEXT=data/iwslt14.tokenized.de-en
$ fairseq preprocess -sourcelang de -targetlang en \
  -trainpref $TEXT/train -validpref $TEXT/valid -testpref $TEXT/test \
  -thresholdsrc 3 -thresholdtgt 3 -destdir data-bin/iwslt14.tokenized.de-en
```
This will write binarized data that can be used for model training to data-bin/iwslt14.tokenized.de-en.

## Training
Use `fairseq train` to train a new model.
Here is the code for training the original fairseq model for the IWSLT14 dataset. 
```
# Fully convolutional sequence-to-sequence model
$ mkdir -p trainings/fconv
$ fairseq train -sourcelang de -targetlang en -datadir data-bin/iwslt14.tokenized.de-en \
  -model fconv -nenclayer 4 -nlayer 3 -dropout 0.2 -optim nag -lr 0.25 -clip 0.1 \
  -momentum 0.99 -timeavg -bptt 0 -savedir trainings/fconv -pretrain

```
We include the file **run-iwlst-de-en-example.sh** as an example of using our DenseNMT architecture for training the IWSLT14 dataset, which gives significant improvement. 

# Generation

We include the file **run-generate-iwlst-de-en-example.sh** as an example of generating predictions and calculating the BLEU scores. 
