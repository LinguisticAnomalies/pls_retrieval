## change the path and file name
ref='.../ref_path'
output='.../hypo_path'
target_file='test.target'
hypo_file='test.hypo'
bert-score -r $ref$target_file -c $output$hypo_file --lang en > ${output}bert_score_$hypo_file
sacrebleu $ref$target_file -i $output$hypo_file -m bleu -b -w 4 > ${output}bleu_$hypo_file
python run_pyrouge.py --target-path $ref --hypo-path ${output} --target-file $target_file --hypo-file $hypo_file
python readability_score.py --target-path $ref --hypo-path ${output} --target-file $target_file --hypo-file $hypo_file
