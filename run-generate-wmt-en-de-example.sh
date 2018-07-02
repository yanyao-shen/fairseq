#!/bin/bash

gpu=0
homepath=~
datadir=~/fairseq/data/wmt-14-en-de


CUDA_VISIBLE_DEVICES=$gpu fairseq generate -denseatt  -sourcelang en -targetlang de -datadir $datadir \
  -dataset test -path $homepath/trainings/fconvdense/model_best.th7 -model fconvdensemopt -beam 5 -batchsize 16 -lenpen 0.1 | tee tmp/gen.$file.out
  
grep ^H tmp/gen.$file.out | cut -f3- | sed 's/@@ //g' > tmp/gen.$file.out.sys
grep ^T tmp/gen.$file.out | cut -f2- | sed 's/@@ //g' > tmp/gen.$file.out.ref
fairseq score -sys tmp/gen.$file.out.sys -ref tmp/gen.$file.out.ref 
