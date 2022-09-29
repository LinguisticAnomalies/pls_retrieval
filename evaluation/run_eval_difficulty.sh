DATA_DIR=src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive
# DATA_DIR=ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_pairs
# TARGET_DIR=/edata/yguo50/plain_language/pls/transformers/examples/research_projects/rag/huggingface_rag/rag_wiki_output_ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/
# TARGET_DIR=/edata/yguo50/plain_language/pls/data/${DATA_DIR}/
TARGET_DIR=/edata/yguo50/plain_language/pls/data/src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive/
python eval_difficulty.py \
    --target-path $TARGET_DIR \
    --target-file 'test.target' > ${TARGET_DIR}/difficulty_target.txt
