export CUDA_VISIBLE_DEVICES=0
MODEL_PATH=./model_path
DATA_PATH=./data_path
mkdir ${MODEL_PATH}input0/
cp ${DATA_PATH}-bin/input0/dict.*  ${MODEL_PATH}input0/
python test.py \
  --model-dir $MODEL_PATH \
  --model-file checkpoint.pt \
  --data-path ${DATA_PATH}-bin/ \
  --bsz 256 \
  --src ${DATA_PATH}/test.input0 \
  --out $MODEL_PATH/test_output.pkl
