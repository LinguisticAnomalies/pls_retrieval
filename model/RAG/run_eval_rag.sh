export CUDA_VISIBLE_DEVICES=1
MODEL_PATH=./model_path
DATA_DIR=./data_dir
python ./transformers/examples/research_projects/rag/eval_rag.py \
    --model_name_or_path $MODEL_PATH \
    --model_type rag_sequence \
    --evaluation_set $DATA_DIR/test.source \
    --gold_data_path $DATA_DIR/test.target \
    --predictions_path $DATA_DIR/test.hypo \
    --eval_mode e2e \
    --gold_data_mode ans \
    --n_docs 5 \
    --max_length 150 \
    --eval_batch_size 32 \
    --recalculate \
