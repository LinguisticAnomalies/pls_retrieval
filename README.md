# Biomedical Lay Language Generation
This repository contains data and models related to the paper: CELLS: A Parallel Corpus for Biomedical Lay Language Generation. 

## Updates
04/24/2023 Upload metadata for CELLS dataset. Title and journal name are available now!

## Datasets
The CELL dataset and the datasets for simplification and background explanation can be found in "./data".

The "xxx.source" files include the scientific text. The "xxx.target" files include the plain language text.
Follow the instructions [here](https://github.com/qiuweipku/Plain_language_summarization) to construct the PubMed dataset for BART pre-training.

## Models
### BART
For BART model, we use the [Fairseq BART](https://github.com/pytorch/fairseq/tree/master/examples/bart) implementation. Download the BART model pretrained on CNN/DM dataset from [here](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz).

Follow the instructions [here](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md) to finetune BART model on CELL data. The hyperparameters for finetuning BART on the plain language generation, simplification and background explanation can be found in "./model/BART/"

### Definition-based explanation retrieval
Definition-based explanation retrieval with UMLS: run "./preprocess/UMLS/umls_ner.py" first to get NERs in the text, then run "./preprocess/UMLS/run_add_umls.sh" to add definitions after the identified NERs. 

Definition-based explanation retrieval with Wikipedia: run "./preprocess/Wiki/run_keywords.sh" first to get the most important words in the text, then sh "./preprocess/Wiki/run_add_wiki.sh" to get definitions from wikipedia after the keywords.

### RAG
For BART model, we use the [Huggingface](https://huggingface.co/docs/transformers/model_doc/rag) implementation. 

Follow the instructions [here](https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag) to finetune BART model on CELL data. The hyperparameters for RAG can be found in "./model/RAG/"


## Checkpoints
Download the models' checkpoints for different tasks [here](https://drive.google.com/drive/u/1/folders/1Qcq93Vo4L8jUD-o06u1z73SqmmPBiWee).
