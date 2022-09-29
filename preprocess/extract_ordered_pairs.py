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

data_dir = '/edata/yguo50/plain_language/pls/data/src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive/'
output_dir = '/edata/yguo50/plain_language/pls/data/extract_ordered_pairs/'
src_path = data_dir + source_name
tgt_path = data_dir + target_name
tgt_txt, src_txt = read_txt(tgt_path), read_txt(src_path)

# tgt_txt, src_txt = tgt_txt[:10], src_txt[:10]
# rouge = Rouge155()
# rouge = Rouge()
# score_dict = {}    
# max_pairs = []
# background_pairs = []
# for paper_i in range(len(tgt_txt)):
#     if paper_i%1000==0:
#         print(paper_i)
#     src_sents = sent_tokenize(re.sub(r'\s+([?.!,:;"])', r'\1', src_txt[paper_i]))
#     tgt_sents = sent_tokenize(re.sub(r'\s+([?.!,:;"])', r'\1', tgt_txt[paper_i]))
#     score_matrix = np.zeros((len(src_sents), len(tgt_sents)))
#     for s_i in range(len(src_sents)):
#         for t_i in range(len(tgt_sents)):
#             score_matrix[s_i, t_i] = rouge.get_scores(src_sents[s_i], tgt_sents[t_i])[0]['rouge-l']['f']
#     score_dict[paper_i] = score_matrix
#     pairs = FindPairs(score_matrix, s_base = 0, t_base = 0)
#     src_bg_list = []
#     tgt_bg_list = []
#     if len(pairs) == 1:
#         src_bg_list = src_sents
#         tgt_bg_list = tgt_sents
#     else:
#         for i in range(0, pairs[1][0]):
#             src_bg_list.append(src_sents[i])
#         for i in range(0, pairs[1][1]):
#             tgt_bg_list.append(tgt_sents[i])

#     src_bg = ' '.join(src_bg_list)
#     tgt_bg = ' '.join(tgt_bg_list)
#     if len(pairs) == 1:
#         background_pairs.append({'index_paper': paper_i, 'src_i': '0_'+str(len(src_sents)), 'tgt_i': '0_'+str(len(tgt_sents)), 
#         'ROUGE': rouge.get_scores(src_bg, tgt_bg)[0]['rouge-l']['f'],
#         'src_sent': src_bg, 'tgt_sent': tgt_bg})
#     else:
#         background_pairs.append({'index_paper': paper_i, 'src_i': '0_'+str(pairs[1][0]-1), 'tgt_i': '0_'+str(pairs[1][1]-1), 
#             'ROUGE': rouge.get_scores(src_bg, tgt_bg)[0]['rouge-l']['f'],
#             'src_sent': src_bg, 'tgt_sent': tgt_bg})

# pd.DataFrame(background_pairs).to_csv(output_dir+'extract_ordered_pairs_background/'+prefix+'.csv', index=False)
# with open(output_dir+ 'extract_ordered_pairs_'+prefix+'_examples_score.pkl', 'wb') as f:
#     pickle.dump(score_dict, f)

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


# data_dir = '/edata/yguo50/plain_language/pls/data/src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive/'
# output_dir = '/edata/yguo50/plain_language/pls/data/extract_ordered_pairs/'
# src_path = data_dir + source_name
# tgt_path = data_dir + target_name
# tgt_txt, src_txt = read_txt(tgt_path), read_txt(src_path)

# rouge = Rouge()
# # rouge = Rouge155()
# score_dict = {}    
# pairs_res = []
# for paper_i in range(len(tgt_txt)):
#     if paper_i % 1000 ==0:
#         print(paper_i)
#     src_sents = sent_tokenize(re.sub(r'\s+([?.!,:;"])', r'\1', src_txt[paper_i]))
#     tgt_sents = sent_tokenize(re.sub(r'\s+([?.!,:;"])', r'\1', tgt_txt[paper_i]))
#     score_matrix = np.zeros((len(src_sents), len(tgt_sents)))
#     for s_i in range(len(src_sents)):
#         for t_i in range(len(tgt_sents)):
#             score_matrix[s_i, t_i] = rouge.get_scores(src_sents[s_i], tgt_sents[t_i])[0]['rouge-l']['f']
#     score_dict[paper_i] = score_matrix
#     pairs = FindPairs(score_matrix, s_base = 0, t_base = 0)
#     # print(pairs)
#     # pairs_res = []
#     # for s_i in range(len(src_sents)):
#     #     for t_i in range(len(tgt_sents)):
#     s_i = 0
#     t_i = 0
#     for i in range(len(pairs)):
#         while s_i < pairs[i][0]:
#             pairs_res.append({'index_paper': paper_i, 'paired': None, 'src_i': s_i, 'tgt_i': None, 'ROUGE': None,
#             'src_sent': src_sents[s_i], 'tgt_sent': None})
#             s_i += 1
#         while t_i < pairs[i][1]:
#             pairs_res.append({'index_paper': paper_i, 'paired': None, 'src_i': None, 'tgt_i': t_i, 'ROUGE': None,
#             'src_sent': None, 'tgt_sent': tgt_sents[t_i]})
#             t_i += 1  
#         pairs_res.append({'index_paper': paper_i, 'paired': 1, 'src_i': pairs[i][0], 'tgt_i': pairs[i][1], 
#             'ROUGE': score_matrix[pairs[i][0]][pairs[i][1]],
#             'src_sent': src_sents[pairs[i][0]], 'tgt_sent': tgt_sents[pairs[i][1]]})
#         s_i += 1
#         t_i += 1  
#     while s_i < len(src_sents):
#         pairs_res.append({'index_paper': paper_i, 'paired': None, 'src_i': s_i, 'tgt_i': None, 'ROUGE': None,
#         'src_sent': src_sents[s_i], 'tgt_sent': None})
#         s_i += 1
#     while t_i < len(tgt_sents):
#         pairs_res.append({'index_paper': paper_i, 'paired': None, 'src_i': None, 'tgt_i': t_i, 'ROUGE': None,
#         'src_sent': None, 'tgt_sent': tgt_sents[t_i]})
#         t_i += 1 
#     pairs_res.append({'index_paper': None, 'paired': None, 'src_i': None, 'tgt_i': None, 'ROUGE': None,
#         'src_sent': None, 'tgt_sent': None})

# pd.DataFrame(pairs_res).to_csv(output_dir+prefix+'.csv', index=False)
