#!/bin/bash

gpu=0
homepath=~
datadir=~/fairseq/data/iwslt14.tokenized.de-en

CUDA_VISIBLE_DEVICES=$gpu fairseq generate -denseatt  -sourcelang en -targetlang de -datadir datadir \
  -dataset test -path $homepath/trainings/fconvdense/model_best.th7 -model fconvdensemopt -beam 10 -batchsize 16  | tee tmp/gen.$file.out

fairseq score -sys tmp/gen.$file.out.sys -ref tmp/gen.$file.out.ref 
