from keybert import KeyBERT
import pandas as pd
from tqdm import tqdm

import argparse
import os

def main():
    """
    Extract keywords from a text.
    python extract_keyword.py \
        --data_path ../data/full_data_extract_elife_annals_medicine_reproductive/ \
        --data_file train.csv \
        --num_keywords 2 \
        --source_name abs_text \
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        help="path importing data file"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        help="file name of the data file"
    )
    parser.add_argument(
        "--num-keywords",
        type=int,
        help="number of keywords to extract",
        default=2
    )
    parser.add_argument(
        "--source-name",
        type=str,
        help="name of the source column",
        default="abs_text"
    )
    args = parser.parse_args()
  
    # Load data
    df = pd.read_csv(args.data_path + args.data_file)
    # # # debug
    # df = df.loc[:2]

    keybert = KeyBERT()
    abs_entity_list = []
    for abs in tqdm(range(len(df))):
        if abs % 1000 == 0:
            print("Processing abstract {}".format(abs))
        abs_text = df[args.source_name][abs]
        keyword = keybert.extract_keywords(abs_text, top_n=args.num_keywords)
        keywords = [i[0] for i in keyword]
        abs_entity_list.append(keywords)
    df['abs_wiki_entity'] = abs_entity_list
    print('saving file to ', args.data_path + 'keywords_' + args.data_file)
    df.to_csv(args.data_path + 'keywords_' + args.data_file, index=False)

if __name__ ==  "__main__":
    main()
        