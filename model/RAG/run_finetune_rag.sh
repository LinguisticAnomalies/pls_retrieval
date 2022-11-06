# Add parent directory to python path to access lightning_base.py
#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="../":"${PYTHONPATH}"
DATA_DIR="./data_dir"
OUTPUT_DIR="./output_dir"
MODEL_NAME_OR_PATH="./model_path/"

python /edata/yguo50/plain_language/pls/transformers/examples/research_projects/rag/finetune_rag.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type rag_sequence \
    --index_name compressed \
    --fp16 \
    --gpus 1 \
    --profile \
    --do_train \
    --do_predict \
    --n_val -1 \
    --train_batch_size 4 \
    --eval_batch_size 2 \
    --max_source_length 200 \
    --max_target_length 150 \
    --val_max_target_length 150 \
    --test_max_target_length 150 \
    --label_smoothing 0.1 \
    --dropout 0.2 \
    --attention_dropout 0.2 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --lr_scheduler polynomial \
    --learning_rate  1e-5 \
    --num_train_epochs 100 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 64 \
    --num_workers 20 \
    > run_finetune_rag.log
##    --max_source_length 1000 \ # wiki: 800 source length + 200 retrieve length

### gradient_accumulation_steps: pairs-256, background-32
