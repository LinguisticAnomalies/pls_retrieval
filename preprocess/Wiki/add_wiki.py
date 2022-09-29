import pandas as pd
from tqdm import tqdm
from nltk.tokenize.punkt import PunktSentenceTokenizer
import json

import argparse
import os

def main():
    """
    Extract keywords from a text.
    python extract_keyword.py \
        --data_path ../data/full_data_extract_elife_annals_medicine_reproductive/ \
        --data_file train.csv \
        --output_path ../data/full_data_extract_elife_annals_medicine_reproductive/add_wiki/ \
        --dict_path ../data/wikimedia/item.csv
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
        "--output-path",
        type=str,
        help="path exporting data file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="output file name"
    )
    parser.add_argument(
        "--dict-path",
        type=str,
        help="path importing dictionary file",
        default="../data/wikimedia/item.csv"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.output_path):
        os.mkdir(args.output_path)
  
    # Load data
    df = pd.read_csv(args.data_path + args.data_file)
    wiki = json.load(open(args.dict_path))
    # # debug
    # df = df.loc[:2]

    abs_wiki_text_list = []
    abs_definition_list = []
    for abs in tqdm(range(len(df))):
        if abs % 1000 == 0:
            print("Processing abstract {}".format(abs))
        abs_text = df['abs_text'][abs]
        keywords = df['abs_wiki_entity'][abs]
        keywords = [i.split("'")[1] for i in keywords.split()]
        # extract desriptions for keywords from wiki
        # print("Extracting wiki description for keywords...")
        descriptions = []
        if len(keywords) > 0:
            for key in keywords:
                if key in wiki:
                    tmp = wiki[key]
                    total = ""
                    for i in range(len(tmp)):
                        if type(tmp[i]) == str:
                            total += key + ' is ' + tmp[i] + '. '
                    descriptions.append(total)
                else:
                    descriptions.append('')
            # add descriptions to text
            # print('add descriptions to text')
            entity = keywords.copy()
            definition = descriptions.copy()
            abs_wiki_text = ''
            for start, end in PunktSentenceTokenizer().span_tokenize(abs_text):
                abs_wiki_text += abs_text[start:end] + ' '
                for e in entity:
                    if e in abs_wiki_text[start:end]:
                            index = entity.index(e)
                            if definition[index] is not None:
                                    abs_wiki_text += definition[index]
                            entity.pop(index)
                            definition.pop(index)
        else:
            abs_wiki_text = ''
            keywords = []
            descriptions = []
        abs_wiki_text_list.append(abs_wiki_text)
        abs_definition_list.append(descriptions)
    df['abs_wiki_text'] = abs_wiki_text_list
    df['abs_wiki_definition'] = abs_definition_list
    print('saving file to ', args.output_path + args.output_file)
    df.to_csv(args.output_path + args.output_file, index=False)

if __name__ ==  "__main__":
    main()