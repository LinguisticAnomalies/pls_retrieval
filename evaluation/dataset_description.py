import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from numpy import random
import textstat
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import pandas as pd
from scipy import stats

def get_read_score(para_list):            
    fkg = []
    gf = []
    cli = []
    dcrs = []
    ari = []
    i = 0
    for p in tqdm(para_list):
        score1 = textstat.flesch_kincaid_grade(p)
        score2 = textstat.gunning_fog(p)
        score3 = textstat.coleman_liau_index(p)
        score4 = textstat.dale_chall_readability_score(p) # Different from other tests, since it uses a lookup table of the most commonly used 3000 English words. T
        score5 = textstat.automated_readability_index(p)
        # score6 = textstat.text_standard(p) # Based upon all the above tests, returns the estimated school grade level required to understand the text.
        fkg.append(score1)
        gf.append(score2)
        cli.append(score3)
        dcrs.append(score4)
        ari.append(score5)
        i += 1
    print('F:{:0.2f}, G:{:0.2f}, C:{:0.2f}, D:{:0.2f}, A:{:0.2f}'.format(
        np.mean(fkg), np.mean(gf), np.mean(cli), np.mean(dcrs), np.mean(ari)))
    return fkg, gf, cli, dcrs, ari

def get_length(para_list):
    length = []
    for p in tqdm(para_list):
        length.append(len(word_tokenize(p)))
    print('Average length: {:0.2f}'.format(np.mean(length)))
    print('Max length: {:0.2f}'.format(np.max(length)))
    print('Min length: {:0.2f}'.format(np.min(length)))
    print('quantile 25%: {:0.2f}'.format(np.percentile(length, 25)))
    print('quantile 50%: {:0.2f}'.format(np.percentile(length, 50)))
    print('quantile 75%: {:0.2f}'.format(np.percentile(length, 75)))
    return length

def get_vocab_size(para_list):
    vocab = set()
    for p in tqdm(para_list):
        for w in word_tokenize(p):
            vocab.add(w)
    print('Vocab size: {}'.format(len(vocab)))
    return vocab

def main():
    """
    Usage::
        python run_pyrouge_train_val_test.py \
            --data-path '/edata/yguo50/plain_language/pls/data/cochrane/' \
            --file-name 'train' \
            --source-name 'abs_text' \
            --target-name 'pls_text' \
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
        type=str,
        help="path importing target file"
    )
    parser.add_argument(
        "--file-name",
        required=True,
        type=str,
        help="path importing hypo file"
    )
    parser.add_argument(
        "--source-name",
        required=True,
        type=str,
        help="path importing target file"
    )
    parser.add_argument(
        "--target-name",
        required=True,
        type=str,
        help="path importing hypo file"
    )

    args = parser.parse_args()
    # df = pd.read_csv(args.data_path + args.file_name + '.csv')

    # source = df[args.source_name].tolist()
    # target = df[args.target_name].tolist()
    with open (args.data_path + args.source_name, 'r') as f:
        source = f.readlines()
    with open (args.data_path + args.target_name, 'r') as f:
        target = f.readlines()

    target_fkg, target_gf, target_cli, target_dcrs, target_ari = get_read_score(target)
    hypo_fkg, hypo_gf, hypo_cli, hypo_dcrs, hypo_ari = get_read_score(source)

    target_length = get_length(target)
    hypo_length = get_length(source)
    

    target_vocab = get_vocab_size(target)
    hypo_vocab = get_vocab_size(source)

    with open(args.data_path + args.file_name + '_readability_socre.txt','w') as f_out:
        f_out.write('Target\n')
        f_out.write('Length: {:0.2f}\n'.format(len(source)))
        f_out.write('Number of tokens: {:0.2f}\n'.format(np.mean(target_length)))
        f_out.write('Vocab size: {}\n'.format(len(target_vocab)))
        f_out.write('F:{:0.2f}, G:{:0.2f}, C:{:0.2f}, D:{:0.2f}, A:{:0.2f}'.format(np.mean(target_fkg), np.mean(target_gf), np.mean(target_cli), np.mean(target_dcrs), np.mean(target_ari)))
        f_out.write('\n')
        f_out.write('-----------------------------------------------------\n')
        f_out.write('Source\n')
        f_out.write('Number of tokens: {:0.2f}\n'.format(np.mean(hypo_length)))
        f_out.write('Vocab size: {}\n'.format(len(hypo_vocab)))
        f_out.write('F:{:0.2f}, G:{:0.2f}, C:{:0.2f}, D:{:0.2f}, A:{:0.2f}'.format(np.mean(hypo_fkg), np.mean(hypo_gf), np.mean(hypo_cli), np.mean(hypo_dcrs), np.mean(hypo_ari)))
        f_out.write('\n')
        f_out.write('ttest_rel F: {}\n'.format(stats.ttest_rel(hypo_fkg, target_fkg)))
        f_out.write('ttest_rel G: {}\n'.format(stats.ttest_rel(hypo_gf, target_gf)))
        f_out.write('ttest_rel C: {}\n'.format(stats.ttest_rel(hypo_cli, target_cli)))
        f_out.write('ttest_rel D: {}\n'.format(stats.ttest_rel(hypo_dcrs, target_dcrs)))
        f_out.write('ttest_rel A: {}\n'.format(stats.ttest_rel(hypo_ari, target_ari)))
        f_out.write('ttest_rel length: {}\n'.format(stats.ttest_rel(hypo_length, target_length)))
if __name__ == "__main__":
    main()