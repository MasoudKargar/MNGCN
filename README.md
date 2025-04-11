# MNGCN
Multistage non-deterministic classification using secondary concept graphs and graph convolutional networks for high-level feature extraction

This is a python implementation of the MNGCN:

* Datasets (Cora, Citeseer, Pubmed)

* Training paradigm for node classification, graph classification, and graph regression tasks

* Visualization

* Evaluation metrics 

### Authors: Masoud Kargar, Nasim Jelodari, Alireza Assadzadeh

## Description

* train.py  
  * main() -- Train a new model for **node classification** task on the *Cora, Citeseer, and Pubmed* datasets
  * evaluate() -- Test the learned model for **node classification** task on the *Cora, Citeseer, and Pubmed* datasets

* MNGCN.py  
  
  * MNGCN() -- model and loss



This sorce Code is for our paper please cite our paper if you use it 

## Citation

@misc{kargar2024multistagenondeterministicclassificationusing,
      title={Multistage non-deterministic classification using secondary concept graphs and graph convolutional networks for high-level feature extraction}, 
      author={Masoud Kargar and Nasim Jelodari and Alireza Assadzadeh},
      year={2024},
      eprint={2411.06212},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.06212}, 
}
