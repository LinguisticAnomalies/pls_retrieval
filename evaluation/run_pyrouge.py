import re
import numpy as np
import random
import pickle
import os
import argparse
from nltk.tokenize import sent_tokenize
from pyrouge import Rouge155

def convert_format(file_dir, save_dir, name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(file_dir,'r+') as f:
        i = 0
        for line in f.readlines():
            sent_list = sent_tokenize(line.strip())
            with open(save_dir + name + '.'+str(i)+'.txt', 'w') as f_out:
                for j in range(len(sent_list)):
                    if j == len(sent_list)-1:
                        f_out.write(sent_list[j])
                    else:
                        f_out.write(sent_list[j]+'\n')
            i += 1

def main():
    """
    Usage::

        python run_pyrouge_new.py \
            --target-path '/edata/yguo50/plain_language/pls/data/cochrane/' \
            --hypo-path '/edata/yguo50/plain_language/pls/output/' \
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
    target_name = "test_target"
    convert_format(args.target_path + args.target_file, args.target_path + target_name + "/", target_name) 

    hypo_name = "test_hypo"
    convert_format(args.hypo_path + args.hypo_file, args.hypo_path + hypo_name + "/", hypo_name)

    r = Rouge155()
    r.system_dir = args.hypo_path + hypo_name + "/"
    r.model_dir = args.target_path + target_name + "/"
    r.system_filename_pattern = hypo_name+'.(\d+).txt'
    r.model_filename_pattern = target_name + '.#ID#.txt'

    output = r.convert_and_evaluate()
    print(output)
    with open(args.hypo_path + args.hypo_file.split('.')[0] + '_pyrouge.txt','w') as f_out:
        f_out.write(output)

if __name__ == "__main__":
    main()
