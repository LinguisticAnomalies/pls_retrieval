export CUDA_VISIBLE_DEVICES=0
MODEL_PATH=/edata/yguo50/pls/output/src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive_ordered_pairs_FixedROUGE/source_target_classification/
DATA_PATH=/edata/yguo50/pls/data/src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive_ordered_pairs_FixedROUGE/source_target_classification
mkdir ${MODEL_PATH}input0/
cp ${DATA_PATH}-bin/input0/dict.*  ${MODEL_PATH}input0/
python test.py \
  --model-dir $MODEL_PATH \
  --model-file checkpoint31.pt \
  --data-path ${DATA_PATH}-bin/ \
  --bsz 256 \
  --src ${DATA_PATH}/test.input0 \
  --out $MODEL_PATH/test_output.pkl