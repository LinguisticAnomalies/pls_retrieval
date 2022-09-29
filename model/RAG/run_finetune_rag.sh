# Add parent directory to python path to access lightning_base.py
#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="../":"${PYTHONPATH}"
DATA_DIR="/edata/yguo50/plain_language/pls/data/ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/"
# DATA_DIR="/edata/yguo50/plain_language/pls/data/ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_pairs/"
OUTPUT_DIR="./rag_wiki_output_ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair_lr_1e-5_dropout_0.2"
# OUTPUT_DIR="./rag_wiki_output_ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_pairs"
MODEL_NAME_OR_PATH="./rag_bart-large-cnn_checkpoint/"
# MODEL_NAME_OR_PATH="/edata/yguo50/plain_language/pls/transformers/examples/research_projects/rag/huggingface_rag/rag_wiki_output_ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_pairs/lr_5e-6_checkpoint2/"
# MODEL_NAME_OR_PATH=/edata/yguo50/plain_language/Plain_language_summarization/checkpoints/pubmed_small_paper_pretrained_bart_large_cnndm/checkpoint_best.pt
# A sample finetuning run, you need to specify data_dir, output_dir and model_name_or_path
# run ./examples/rag/finetune_rag.sh --help to see all the possible options

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

# DATA_DIR="./full_data_extract_elife_annals_medicine_reproductive/"