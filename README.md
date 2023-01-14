# Introduction

This is the source code of our TOMM 2023 paper "MKVSE: Multimodal Knowledge Enhanced Visual-Semantic Embedding for Image-Text Retrieval". Please cite the following paper if you use our code.

Duoduo Feng, Xiangteng He and Yuxin Peng, "MKVSE: Multimodal Knowledge Enhanced Visual-Semantic Embedding for Image-Text Retrieval", ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 2023.




# Dependencies

We referred to the implementations of [GPO](https://github.com/woodfrog/vse_infty) and [CVSE](https://github.com/BruceW91/CVSE) to build up our codebase. We used the following key dependencies:

- Python 3.7.3

- Pytorch 1.2.0

- Transformers 4.12.5

Run ``` conda env create -f environment.yml || conda env update -f environment.yml``` to install the same dependencies as our experiments. Download the pre-trained Bert params "config.json", "pytorch_model.bin" and "vocab.txt" in hugging face's [bert-base-uncased](https://huggingface.co/bert-base-uncased/tree/main) and put them in the folder MKG/bert-base-uncased and MGCN/bert-base-uncased.




# Data Preparation
We organize the data folder in the following manner:
```
data
├── coco
│   ├── precomp  # pre-computed BUTD region features for COCO
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   ├── Concept_annotaions  # graph data from CVSE
│   │      ├── coco_adj_concepts.pkl
│   │      └── coco_concepts_glove_word2vec.pkl
├── f30k
│   ├── precomp  # pre-computed BUTD region features for Flickr30k
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   ├── Concept_annotaions  # graph data from CVSE
│   │      ├── f30k_adj_concepts.pkl
│   │      └── f30k_concepts_glove_word2vec.pkl
└── VisualGenome
       ├── image_data.json # image meta data
       └── objects.json # object data
```

The data preparation steps are as follows:
1. Download the Flickr30K/MSCOCO precomputed BUTD features and corresponding vocabularies are from the offical repo of [BUTD](https://github.com/peteanderson80/bottom-up-attention) and put them in the folder data/f30k/precomp and data/coco/precomp.
2. Download the files "image meta data" and "objects"  of [VisualGenome v1.2](http://visualgenome.org/api/v0/api_home.html). Then unzip and put them in the folder data/VisualGenome.
3. Download the files "f30k_concepts_glove_word2vec.pkl" and "f30k_adj_concepts.pkl" of [CVSE](https://github.com/BruceW91/CVSE) and put them in the folder data/f30k/Concept_annotations/. Download the files "coco_concepts_glove_word2vec.pkl" and "coco_adj_concepts.pkl" of [CVSE](https://github.com/BruceW91/CVSE) and put them in the folder data/coco/Concept_annotations/.  



# Usage

Start training by executing the following commands. ```<this_project_abspath>``` is the absolute path of this project. It will train the ```<model_name> (MKG/MGCN)``` model  on the ``` <dataset_name> (f30k/coco)``` dataset.
```
export PROJECT_PATH="<this_project_abspath>"
cd ${PROJECT_PATH}/<model_name>
bash train_<model_name>_<dataset_name>.sh
```

To get the ensemble results of two datasets, run the following scripts:
```
export PROJECT_PATH="<this_project_abspath>"
bash eval_ensemble.sh
```


For any questions, feel free to contact us (fengduoduo@pku.edu.cn). Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl/home/) for more information about our papers, source codes, and datasets.
