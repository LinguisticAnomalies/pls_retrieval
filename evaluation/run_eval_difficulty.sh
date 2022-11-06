DATA_DIR=../hypo_path
TARGET_DIR=../target_path
python eval_difficulty.py \
    --target-path $TARGET_DIR \
    --target-file 'test.target' > ${TARGET_DIR}/difficulty_target.txt
