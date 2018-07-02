mkdir -p trainings/fconvdense

CUDA_VISIBLE_DEVICES=0 fairseq train -sourcelang de -targetlang en \
  -datadir data-bin/iwslt14.tokenized.de-en -model fconvdensemopt \
  -nenclayer 8 -nlayer 8 -dropout 0.2 -optim nag -lr 0.25 -clip 0.1 \
  -momentum 0.99 -timeavg -bptt 0 -savedir trainings/fconvdense \
  -multistepfinetune -opt 1 -optenc 1 -maxbatch 1200 -batchsize 16 \
  -denseatt -blocklength 6 -nhid 192 -nhiddec 192 -nhid_acc 96 -nhid_acc_dec 96 -pretrain

