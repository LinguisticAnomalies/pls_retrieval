export CUDA_VISIBLE_DEVICES=1
python /edata/yguo50/plain_language/pls/transformers/examples/research_projects/rag/eval_rag.py \
    --model_name_or_path ./rag_wiki_output_ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_pairs/checkpoint5 \
    --model_type rag_sequence \
    --evaluation_set /edata/yguo50/plain_language/pls/data/ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_pairs/test.source \
    --gold_data_path /edata/yguo50/plain_language/pls/data/ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_pairs/test.target \
    --predictions_path ./rag_wiki_output_ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_pairs/e2e_preds_checkpoint5.txt \
    --eval_mode e2e \
    --gold_data_mode ans \
    --n_docs 5 \
    --max_length 150 \
    --eval_batch_size 32 \
    --recalculate \

# python /edata/yguo50/plain_language/pls/transformers/examples/research_projects/rag/eval_rag.py \
#     --model_name_or_path ./rag_wiki_output_ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/checkpoint6/ \
#     --model_type rag_sequence \
#     --evaluation_set /edata/yguo50/plain_language/pls/data/ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/test.source \
#     --gold_data_path /edata/yguo50/plain_language/pls/data/ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/test.target \
#     --predictions_path ./rag_wiki_output_ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/e2e_preds_checkpoint6.txt \
#     --eval_mode e2e \
#     --gold_data_mode ans \
#     --n_docs 5 \
#     --max_length 150 \
#     --eval_batch_size 32 \
#     --recalculate \

# python /edata/yguo50/plain_language/pls/transformers/examples/research_projects/rag/eval_rag.py \
#     --model_name_or_path ./rag_wiki_output_ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/checkpoint7/ \
#     --model_type rag_sequence \
#     --evaluation_set /edata/yguo50/plain_language/pls/data/ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/test.source \
#     --gold_data_path /edata/yguo50/plain_language/pls/data/ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/test.target \
#     --predictions_path ./rag_wiki_output_ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/e2e_preds_checkpoint7.txt \
#     --eval_mode e2e \
#     --gold_data_mode ans \
#     --n_docs 5 \
#     --max_length 150 \
#     --eval_batch_size 32 \
#     --recalculate \

# python /edata/yguo50/plain_language/pls/transformers/examples/research_projects/rag/eval_rag.py \
#     --model_name_or_path ./rag_wiki_output_ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/checkpoint8/ \
#     --model_type rag_sequence \
#     --evaluation_set /edata/yguo50/plain_language/pls/data/ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/test.source \
#     --gold_data_path /edata/yguo50/plain_language/pls/data/ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/test.target \
#     --predictions_path ./rag_wiki_output_ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/e2e_preds_checkpoint8.txt \
#     --eval_mode e2e \
#     --gold_data_mode ans \
#     --n_docs 5 \
#     --max_length 150 \
#     --eval_batch_size 32 \
#     --recalculate \

# python /edata/yguo50/plain_language/pls/transformers/examples/research_projects/rag/eval_rag.py \
#     --model_name_or_path ./rag_wiki_output_ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/checkpoint4/ \
#     --model_type rag_sequence \
#     --evaluation_set /edata/yguo50/plain_language/pls/data/ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/test.source \
#     --gold_data_path /edata/yguo50/plain_language/pls/data/ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/test.target \
#     --predictions_path ./rag_wiki_output_ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/e2e_preds_checkpoint4.txt \
#     --eval_mode e2e \
#     --gold_data_mode ans \
#     --n_docs 5 \
#     --max_length 150 \
#     --eval_batch_size 32 \
#     --recalculate \

    # --print_predictions \
    # --n_docs 5 \ # You can experiment with retrieving different number of documents at evaluation time
    # --recalculate \ # adding this parameter will force recalculating predictions even if predictions_path already exists
    #  export CUDA_VISIBLE_DEVICES=0
