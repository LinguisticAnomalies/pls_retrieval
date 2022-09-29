export CUDA_VISIBLE_DEVICES=1
PREFIX=/edata/yguo50/plain_language/pls/data/ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_pairs
DATA_DIR=add_wiki
MODEL_PATH=/edata/yguo50/plain_language/pls/output/pairs_lr_3e-05_${DATA_DIR}_bart_large_cnn
DATA_PATH=$PREFIX/$DATA_DIR
LENPEN=1.0
mkdir -p $MODEL_PATH/lenpen_$LENPEN
cp ${DATA_PATH}/bin/dict.*  $MODEL_PATH
python examples/bart/summarize.py \
  --model-dir $MODEL_PATH \
  --model-file checkpoint_best.pt \
  --lenpen $LENPEN \
  --bsz 64 \
  --max-len 150 \
  --min-len 0 \
  --src ${DATA_PATH}/test.source \
  --out $MODEL_PATH/lenpen_$LENPEN/test.hypo