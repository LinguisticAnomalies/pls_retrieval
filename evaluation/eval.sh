CUDA_VISIBLE_DEVICES=1 \
# ref='/edata/yguo50/plain_language/pls/data/ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/'
ref='/edata/yguo50/plain_language/pls/data/ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_pairs/'
# ref='/edata/yguo50/plain_language/pls/data/src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive/'
# output='/edata/yguo50/plain_language/pls/transformers/examples/research_projects/rag/huggingface_rag/rag_wiki_output_ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/'
output='/edata/yguo50/plain_language/pls/transformers/examples/research_projects/rag/huggingface_rag/rag_wiki_output_ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_pairs/'
# output='/edata/yguo50/plain_language/pls/output/pairs_lr_3e-05_add_wiki_bart_large_cnn/lenpen_1.0/'
# output='/edata/yguo50/plain_language/pls/output/lr_3e-05_ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair_bart_large_cnn/lenpen_1.0/'
# output='/edata/yguo50/plain_language/pls/data/ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_background_before_2ndPair/'
# output='/edata/yguo50/plain_language/pls/data/ordered_pairs_src_200_tgt_150_rouge_0.1-0.4_FixedROUGE_pairs/'
# output='/edata/yguo50/plain_language/pls/data/src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive/'
# output='/edata/yguo50/plain_language/pls/output/lr_3e-05_src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive_bart_large_cnn/lenpen_1.0/'
target_file='test.target'
# hypo_file='whole.source'
# hypo_file='test.hypo'
hypo_file='e2e_preds_checkpoint5.txt'
bert-score -r $ref$target_file -c $output$hypo_file --lang en > ${output}bert_score_$hypo_file
# sacrebleu $ref$target_file -i $output$hypo_file -m bleu -b -w 4 > ${output}bleu_$hypo_file
# sacrebleu $ref$target_file -i $output$hypo_file -m bleu > ${output}bleu_$hypo_file
# python run_pyrouge.py --target-path $ref --hypo-path ${output} --target-file $target_file --hypo-file $hypo_file
# python readability_score.py --target-path $ref --hypo-path ${output} --target-file $target_file --hypo-file $hypo_file
# python bleu_score.py --target-path $ref --hypo-path $output --target-file $target_file --hypo-file $hypo_file > ${output}bleu_score.txt
