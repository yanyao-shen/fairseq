#!/bin/bash
gpu=0
homepath=/home
target=$homepath/trainings/fconvdense-wmt
bindataset=$homepath/fairseq-data/data-bin/wmt-14-en-de
mkdir -p target
CUDA_VISIBLE_DEVICES=$gpu fairseq train -sourcelang en -targetlang de -datadir $bindataset \
  -model fconvdensemopt -nenclayer 15 -nlayer 15 -dropout 0.1 -optim nag -lr 0.25 -clip 0.1 \
  -momentum 0.99 -timeavg -bptt 0 -savedir $target -pretrain \
  -fconv_nhids 512,512,512,512,512,512,512,512,512,512,1024,1024,1024,2048,2048 \
  -fconv_kwidths 3,3,3,3,3,3,3,3,3,3,3,3,3,1,1 \
  -fconv_klmwidths 3,3,3,3,3,3,3,3,3,3,3,3,3,1,1 \
  -fconv_nlmhids 512,512,512,512,512,512,512,512,512,512,1024,1024,1024,2048,2048 \
  -fconv_nhid_accs 128,128,128,128,128,128,128,128,128,128,256,256,256,512,512 \
  -fconv_nhid_accs_dec 128,128,128,128,128,128,128,128,128,128,256,256,256,512,512 \
  -maxbatch 1200 \
  -multistepfinetune -seed 1111 -opt 1 -optenc 1 -ninembed 768 -noutembed 768 -nembed 768 -minepochtoanneal 20 -batchsize 48 -ndatathreads 4 -log -denseatt -blocklength 6 > $homepath/logs/log_$target.log


