from rouge import Rouge
# from pyrouge import Rouge155
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import argparse
import json
import os
import numpy as np
import pickle
import re

prefix = 'val'
source_name = prefix+'.source'
target_name = prefix+'.target'

def read_txt(path):
    f = open(path, 'r')
    content = f.readlines()
    return content

def FindPairs(s2t_all, s_base, t_base):
	ns, nt = np.shape(s2t_all)
	if ns==0 or nt==0:
		return []
	maxs,maxt,maxv = -1, -1, -100
	for i in range(ns):
		for j in range(nt):
			if s2t_all[i,j] > maxv:
				maxv = s2t_all[i,j]
				maxs = i
				maxt = j
	# print (maxs, maxt, maxv)
	s2t_small = s2t_all[:maxs, :maxt]
	s2t_large = s2t_all[(maxs+1):, (maxt+1):]
	pairs_all = []
	pairs_all.extend(FindPairs(s2t_small, s_base, t_base))
	pairs_all.extend([[s_base + maxs,t_base + maxt]])
	pairs_all.extend(FindPairs(s2t_large, s_base + maxs + 1, t_base + maxt + 1))
	return pairs_all

data_dir = './data_dir/'
output_dir = './output_dir/'
src_path = data_dir + source_name
tgt_path = data_dir + target_name
tgt_txt, src_txt = read_txt(tgt_path), read_txt(src_path)

rouge = Rouge()
score_dict = {}    
max_pairs = []
background_pairs = []
for paper_i in range(len(tgt_txt)):
    if paper_i%1000==0:
        print(paper_i)
    src_sents = sent_tokenize(re.sub(r'\s+([?.!,:;"])', r'\1', src_txt[paper_i]))
    tgt_sents = sent_tokenize(re.sub(r'\s+([?.!,:;"])', r'\1', tgt_txt[paper_i]))
    score_matrix = np.zeros((len(src_sents), len(tgt_sents)))
    for s_i in range(len(src_sents)):
        for t_i in range(len(tgt_sents)):
            score_matrix[s_i, t_i] = rouge.get_scores(src_sents[s_i], tgt_sents[t_i])[0]['rouge-l']['f']
    score_dict[paper_i] = score_matrix
    pairs = FindPairs(score_matrix, s_base = 0, t_base = 0)
    src_bg_list = []
    tgt_bg_list = []
    for i in range(0, pairs[0][0]+1):
        src_bg_list.append(src_sents[i])
    for i in range(0, pairs[0][1]+1):
        tgt_bg_list.append(tgt_sents[i])

    src_bg = ' '.join(src_bg_list)
    tgt_bg = ' '.join(tgt_bg_list)
    
    background_pairs.append({'index_paper': paper_i, 'src_i': '0_'+str(pairs[0][0]), 'tgt_i': '0_'+str(pairs[0][1]), 
        'ROUGE': rouge.get_scores(src_bg, tgt_bg)[0]['rouge-l']['f'],
        'src_sent': src_bg, 'tgt_sent': tgt_bg})

pd.DataFrame(background_pairs).to_csv(output_dir+'extract_ordered_pairs_background/'+prefix+'.csv', index=False)
with open(output_dir+ 'extract_ordered_pairs_'+prefix+'_examples_score.pkl', 'wb') as f:
    pickle.dump(score_dict, f)
