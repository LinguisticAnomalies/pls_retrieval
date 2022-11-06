wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

TASK=../data_path
for SPLIT in train val test
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$TASK/$SPLIT.$LANG" \
    --outputs "$TASK/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/val.bpe" \
  --testpref "${TASK}/test.bpe" \
  --destdir "${TASK}/bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
