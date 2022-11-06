export CUDA_VISIBLE_DEVICES=1
MODEL_PATH=../model_path
DATA_PATH=../data_path
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
