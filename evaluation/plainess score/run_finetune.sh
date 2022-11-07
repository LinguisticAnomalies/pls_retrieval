TOTAL_NUM_UPDATES=500000  # 10 epochs through IMDB for bsz 32
WARMUP_UPDATES=500      # 6 percent of the number of updates
LR=1e-6                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2           # Number of classes for the classification task.
MAX_SENTENCES=16         # Batch size.
UPDATE_FREQ=8
ROBERTA_PATH=./roberta.large/model.pt
SAVE_DIR=./save_dir
mkdir -p $SAVE_DIR


CUDA_VISIBLE_DEVICES=0 fairseq-train ./data_dir \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --task sentence_prediction \
    --num-classes $NUM_CLASSES \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction \
    --dropout 0.2 --attention-dropout 0.2 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --max-epoch 1000 \
    --best-checkpoint-metric accuracy \
    --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --update-freq $UPDATE_FREQ \
    --save-dir $SAVE_DIR \
    > $SAVE_DIR/train.log
