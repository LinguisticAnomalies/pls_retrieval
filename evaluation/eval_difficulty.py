import pandas as pd
import argparse
from tqdm import tqdm
import nltk
from scipy import stats
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

def get_tags(abs_text, tag1, tag2, adj_tag, adv_tag, num_tag, conj_tag):
    """ Return number of verbs, nouns, adjectives, adverbs, numbers, and conjuction words"""
    vb_count = []
    nn_count = []
    adj_count = []
    adv_count = []
    num_count = []
    conj_count = []
    for i in range(len(abs_text)):
        text = nltk.word_tokenize(abs_text[i])
        pos_tagged = nltk.pos_tag(text)
        vb_count.append(len([x for x in pos_tagged if x[1][0] == tag1])/len(text))
        nn_count.append(len([x for x in pos_tagged if x[1][0] == tag2])/len(text))
        adj_count.append(len([x for x in pos_tagged if x[1] == adj_tag])/len(text))
        adv_count.append(len([x for x in pos_tagged if x[1] == adv_tag])/len(text))
        num_count.append(len([x for x in pos_tagged if x[1] == num_tag])/len(text))
        conj_count.append(len([x for x in pos_tagged if x[1] == conj_tag])/len(text))
    return vb_count, nn_count, adj_count, adv_count, num_count, conj_count

def get_length(para_list):
    """ Return paragraph length in token level"""
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
    """ Return vocab size of the paragraph"""
    vocab = set()
    for p in tqdm(para_list):
        for w in word_tokenize(p):
            vocab.add(w)
    print('Vocab size: {}'.format(len(vocab)))
    return vocab

def get_familiarity(abs_text, word_freq_dict):
    """ Return word familiarity"""
    abs_lexical_familiarity = []
    for i in range(len(abs_text)):
        abs_words = nltk.word_tokenize(abs_text[i].lower())
        abs_words = [x for x in abs_words if x.isalnum()]
        single_familiarty = []
        for x in abs_words:
            if x in word_freq_dict.keys():
                single_familiarty.append(word_freq_dict[x])
        abs_lexical_familiarity.append(sum(single_familiarty)/len(single_familiarty))
    return abs_lexical_familiarity


def main():
    """
    Return pos of tag, length of paragraph, word familiarity, and vocab size of hypo and target texts
    Usage::
        python eval_difficulty.py \
            --target-path '../target_path/' \
            --target-file 'test.target' \
            --hypo-path '../hypo_path/' \
            --hypo-file 'test.hypo' \
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-path",
        required=True,
        type=str,
        help="path importing target file"
    )
    parser.add_argument(
        "--target-file",
        required=True,
        type=str,
        help="path importing hypo file"
    )


    args = parser.parse_args()
    with open (args.target_path + args.target_file) as f:
        tgt_text = f.readlines()

    
    # tgt_text = tgt_text[:10]
    ## number of verbs and nouns
    nltk.download('averaged_perceptron_tagger')
    tgt_vb, tgt_nn, tgt_adj, tgt_adv, tgt_num, tgt_conj = get_tags(tgt_text, 'V', 'N', 'JJ', 'RB', 'CD', 'CC')
    
    ## length of sentences
    print('Length of paragraphs')
    tgt_length = get_length(tgt_text)


    ## vocab size
    print('Vocab size')
    tgt_vocab = get_vocab_size(tgt_text)


    ## word familiarity
    print('Word familiarity')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    word_freq = pd.read_csv('./wikiidf_terms.csv')
    word_freq_dict = {}
    for i in range(len(word_freq)):
        word_freq_dict[word_freq['token'][i]] = word_freq['idf'][i]

    tgt_familiarity = get_familiarity(tgt_text, word_freq_dict)


    df_output = pd.DataFrame({'tgt_vb': tgt_vb, 'tgt_nn': tgt_nn, 'tgt_adj': tgt_adj, 'tgt_adv': tgt_adv, 'tgt_num': tgt_num, 'tgt_conj': tgt_conj,
                     'IDF': tgt_familiarity, 'length': tgt_length})
    df_output.to_csv(args.target_path + 'diff_count_target.csv')

    
if __name__ == "__main__":
    main()
