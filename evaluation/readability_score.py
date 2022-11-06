import numpy as np
from numpy import random
import textstat
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import pickle

def get_read_score(para_list):            
    fkg = []
    gf = []
    cli = []
    dcrs = []
    ari = []
    i = 0
    for p in tqdm(para_list):
        if i % 1000 == 0:
            print(i)
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

def main():
    """
    Usage::

        python run_pyrouge_new.py \
            --target-path '../target_path' \
            --hypo-path '../hypo_path' \
            --target-file 'test.target' \
            --hypo-file 'test.hypo'
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-path",
        required=True,
        type=str,
        help="path importing target file"
    )
    parser.add_argument(
        "--hypo-path",
        required=True,
        type=str,
        help="path importing hypo file"
    )
    parser.add_argument(
        "--target-file",
        required=True,
        type=str,
        help="target file name"
    )
    parser.add_argument(
        "--hypo-file",
        required=True,
        type=str,
        help="hypo file name"
    )


    args = parser.parse_args()
    with open (args.target_path + args.target_file) as f:
        target = f.readlines()
    with open (args.hypo_path + args.hypo_file) as f:
        hypo = f.readlines()

    target_fkg, target_gf, target_cli, target_dcrs, target_ari = get_read_score(target)
    hypo_fkg, hypo_gf, hypo_cli, hypo_dcrs, hypo_ari = get_read_score(hypo)

    target_score = {}
    target_score['fkg'] = target_fkg
    target_score['gf'] = target_gf
    target_score['cli'] = target_cli
    target_score['dcrs'] = target_dcrs
    target_score['ari'] = target_ari
    pickle.dump(target_score, open(args.hypo_path+args.hypo_file.split('.')[0]+'_readability_score_target.pkl', 'wb'))

    hypo_score = {}
    hypo_score['fkg'] = hypo_fkg
    hypo_score['gf'] = hypo_gf
    hypo_score['cli'] = hypo_cli
    hypo_score['dcrs'] = hypo_dcrs
    hypo_score['ari'] = hypo_ari
    pickle.dump(hypo_score, open(args.hypo_path+args.hypo_file.split('.')[0]+'_readability_score_hypo.pkl', 'wb'))


    with open(args.hypo_path + args.hypo_file.split('.')[0] + '_readability_score.txt','w') as f_out:
        f_out.write('Target\n')
        f_out.write('F:{:0.2f}, G:{:0.2f}, C:{:0.2f}, D:{:0.2f}, A:{:0.2f}'.format(np.mean(target_fkg), np.mean(target_gf), np.mean(target_cli), np.mean(target_dcrs), np.mean(target_ari)))
        f_out.write('\n')
        f_out.write('Hypo\n')
        f_out.write('F:{:0.2f}, G:{:0.2f}, C:{:0.2f}, D:{:0.2f}, A:{:0.2f}'.format(np.mean(hypo_fkg), np.mean(hypo_gf), np.mean(hypo_cli), np.mean(hypo_dcrs), np.mean(hypo_ari)))
if __name__ == "__main__":
    main()
