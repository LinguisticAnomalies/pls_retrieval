# CELLS: A Parallel Corpus for Biomedical Lay Language Generation
This repository contains data and models related to the paper: CELLS: A Parallel Corpus for Biomedical Lay Language Generation. 

In particular, 

## Data Preprocess
1. Datasets
2. Data-augmentation with UMLS and Wikipedia
Augmented with UMLS: run umls_ner.py first to get NERs in the text, then run run_add_umls.sh to add definitions after the identified NERs.
Augmented with Wikipedia: run run_keywords.sh first to get the most important words in the text, then sh run_add_wiki.sh to get definitions from wikipedia after the keywords.

## Models

## Evaluation
