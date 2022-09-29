import json
import pandas as pd
import numpy as np
import spacy
import scispacy
from scispacy.linking import EntityLinker
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"]="1"
prefix = "val"

data_source_path = '/homes/gws/wqiu0528/medical_plain/paragraph_level/'+prefix+'.source'
# data_target_path = '/homes/gws/wqiu0528/medical_plain/paragraph_level/'+prefix+'.target'
save_path = '/homes/gws/wqiu0528/medical_plain/paragraph_level/'+prefix+'_umls.json'

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def get_umls_dict(text):
    res = {}
    doc = nlp(text)
    num = 0
    for i in range(len(doc.ents)):
        entity = doc.ents[i]
        if len(entity._.kb_ents) == 0:
            continue
        entity_list = []
        for umls_ent in entity._.kb_ents:
            entity_list.append(linker.kb.cui_to_entity[umls_ent[0]])
        umls_dict = pd.DataFrame(entity_list).iloc[0].to_dict()
        umls_dict['text'] = entity.text
        umls_dict['start_char'] = int(entity.start_char)
        umls_dict['end_char'] = int(entity.end_char)
        umls_dict['label_'] = entity.label_
        res[num] = umls_dict
        num += 1
    return res

nlp = spacy.load("en_core_sci_scibert")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls", "k": 1})
linker = nlp.get_pipe("scispacy_linker")

source = [s.strip() for s in open(data_source_path, 'r').readlines()]
# target = [s.strip() for s in open(data_target_path, 'r').readlines()]

umls_res = {}

for idx in tqdm(range(len(source))):
    umls_res[idx] = {}
    umls_res[idx]['abs_umls'] = get_umls_dict(source[idx])
    # umls_res[idx]['pls_umls'] = get_umls_dict(target[idx])

with open(save_path, 'w') as json_file:
    json.dump(umls_res, json_file, cls=NpEncoder)