TOTAL_NUM_UPDATES=500000  # 10 epochs through IMDB for bsz 32
WARMUP_UPDATES=500      # 6 percent of the number of updates
LR=1e-6                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2           # Number of classes for the classification task.
MAX_SENTENCES=16         # Batch size.
UPDATE_FREQ=8
ROBERTA_PATH=./roberta.large/model.pt
SAVE_DIR=/edata/yguo50/pls/output/src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive_ordered_pairs_FixedROUGE/source_target_classification_10_150/
mkdir -p $SAVE_DIR


CUDA_VISIBLE_DEVICES=0 fairseq-train /edata/yguo50/pls/data/src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive_ordered_pairs_FixedROUGE/source_target_classification_10_150-bin \
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



        # --max-tokens 4400 \
    # --regression-target \
            # --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
                # --num-classes $NUM_CLASSES \
        # SAVE_DIR=/edata/yguo50/output/src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive_ordered_pairs/sentence_classification_binary/
    # --criterion cross_entropy
    #     --no-epoch-checkpoints \
    # --best-checkpoint-metric loss \