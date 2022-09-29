TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=32

BART_PATH=./bart_model/bart.large.cnn/model.pt
# BART_PATH=/edata/yguo50/plain_language/Plain_language_summarization/checkpoints/pubmed_small_paper_pretrained_bart_large_cnndm/checkpoint_best.pt
PREFIX=/edata/yguo50/plain_language/pls/data/full_data_extract_elife_annals_medicine_reproductive/
DATA_DIR=add_umls
SAVE_DIR=/edata/yguo50/plain_language/pls/output/paragraph_lr_${LR}_${DATA_DIR}_bart_large_cnn/
mkdir -p $SAVE_DIR
CUDA_VISIBLE_DEVICES=0 fairseq-train  $PREFIX/$DATA_DIR/bin/ \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters\
    --no-epoch-checkpoints \
    --save-dir $SAVE_DIR \
    > $SAVE_DIR/train.log


