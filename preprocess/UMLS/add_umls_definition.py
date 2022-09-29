import json
import pandas as pd
from nltk.tokenize.punkt import PunktSentenceTokenizer
import os
import argparse
from tqdm import tqdm

def get_umls_entity(temp, entity_type, var):
    abs_definition = []
    abs_entity = []
    for key, value in temp[var].items():  
        # if value['types'][0] in entity_type:
        if value['text'] not in abs_entity:
            abs_entity.append(value['text'])
            abs_definition.append(value['definition'])
    if var == 'abs_umls':
        temp['abs_entity'] = abs_entity
        temp['abs_definition'] = abs_definition
    else:
        temp['pls_entity'] = abs_entity
        temp['pls_definition'] = abs_definition
    return temp

def main():
    """
    Usage::
        python add_umls_definition.py \
            --data-path ../MedLane_data/ \
            --save-path ../MedLane_data/ \
            --umls-file test_umls.json \
            --csv-file test.csv

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        help="path importing data file",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="path importing data file",
    )
    
    parser.add_argument(
        "--umls-file",
        required=True,
        type=str,
        help="json file that contains umls data",
    )

    parser.add_argument(
        "--csv-file",
        required=True,
        type=str,
        help="csv file",
    )

    
    args = parser.parse_args()
    
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    
    data = json.load(open(args.data_path + args.umls_file))
    df = pd.read_csv(args.data_path + args.csv_file)

    entity_type = ['T047', 'T184', 'T060', 'T121'] # change based on your needs
    #https://www.nlm.nih.gov/research/umls/sourcereleasedocs/current/MSHNOR/stats.html
    # T047: Disease or syndrome
    # T184: Sign or symptom
    # T060: Diagnostic procedure
    # T121: Pharmacologic substance
    print('Start to extract specific umls entity...')
    for i in tqdm(range(len(data))):
        data[str(i)] = get_umls_entity(data[str(i)], entity_type, 'abs_umls')
        data[str(i)] = get_umls_entity(data[str(i)], entity_type, 'pls_umls')
    
    print("Start to add umls definition to csv file...")
    abs_umls_text_list = []
    abs_entity_list = []
    pls_entity_list = []
    for i in tqdm(range(len(data))):
        abs_umls_text = ''
        entity = data[str(i)]['abs_entity'].copy()
        definition = data[str(i)]['abs_definition'].copy()
        abs_text = df['abs_text'][i]
        for start, end in PunktSentenceTokenizer().span_tokenize(abs_text):
                abs_umls_text += abs_text[start:end] + ' '
                for e in entity:
                    if e in abs_text[start:end]:
                            index = entity.index(e)
                            if definition[index] is not None:
                                    abs_umls_text += definition[index]
                            entity.pop(index)
                            definition.pop(index)
        abs_umls_text_list.append(abs_umls_text)
        abs_entity_list.append(entity)
        pls_entity_list.append(data[str(i)]['pls_entity'])
    df['abs_umls_text'] = abs_umls_text_list
    df['abs_entity'] = abs_entity_list
    df['pls_entity'] = pls_entity_list
    df.to_csv(args.save_path + args.csv_file, index=False)
    
    with open(args.save_path + args.umls_file, 'w') as f:
        json.dump(data, f)

if __name__ ==  "__main__":
    main()