# Biomedical Lay Language Generation
This repository contains data and models related to the paper: [Retrieval augmentation of large language models for lay language generation](https://www.sciencedirect.com/science/article/pii/S1532046423003015?dgcid=author). This paper was arxived as CELLS: A Parallel Corpus for Biomedical Lay Language Generation. 

## Updates
01/18/2024 Paper was accepted by Journal of Biomedical Informatics. Check the latest version with GPT-4 and Llama-2 here: [Retrieval augmentation of large language models for lay language generation](https://www.sciencedirect.com/science/article/pii/S1532046423003015?dgcid=author)

05/01/2023 Update wiki_dict.json file! Click [here](https://drive.google.com/file/d/1c0nz577gaghbP0GlRfRhOy8EyuSdpHZO/view?usp=share_link) to download it.

04/24/2023 Upload metadata for CELLS dataset. Title and journal name are available now!

## Datasets
Datasets can be found in "./data". The "xxx.source" files include the scientific text, while the "xxx.target" files include the plain language text. Follow the instructions [here](https://github.com/qiuweipku/Plain_language_summarization) to construct the PubMed dataset for BART pre-training.

Datasets for different applications (details can be found in 3.1.2 Dataset applications):
- CELLS: The paragraph-paired data of scientific abstracts and plain language summaries for the lay language generation task.
- BELLS: The paragraph segment-paired data for background explanations.
- SELLS: The sentence-level paired data for simplification.
- Validated dataset: Randomly sampled data annotated by annotators for background explanations and simplification.

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

### LLMs
To evaluate the performance of LLMs in generating background explanations or plain language summaries, we utilized Llama 2 (Llama-2-70B-chat) and GPT-4 (accessed in September 2023). We explored two prompts:

- "Summarize in plain language: input"
- "Summarize in plain language, providing necessary explanations: input"

To further assess the impact of the retrieval-augmented approach on LLMs, we established two settings for input:

- The source alone
- The source combined with Wikipedia definitions as identified using KeyBERT

The generation process was configured with a maximum length of 150 tokens. All other parameters were set to their default values.


## Checkpoints
Download the models' checkpoints for different tasks [here](https://drive.google.com/drive/u/1/folders/1Qcq93Vo4L8jUD-o06u1z73SqmmPBiWee).
