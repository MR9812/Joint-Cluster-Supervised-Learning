## Joint-Cluster Supervised Learning Framework

This is the official implementation of the following paper:

> [Rethinking Independent Cross-Entropy Loss For Graph-Structured Data](https://arxiv.org/pdf/2405.15564)
> 
> Accepted by ICML 2024

In this work, we propose a new framework, termed joint-cluster supervised learning, to model the joint distribution of each node with its corresponding cluster. We learn the joint distribution of node and cluster labels conditioned on their representations, and train GNNs with the obtained joint loss. In this way, the data label reference signals extracted from the local cluster explicitly strengthen the discrimination ability on the target node. 

## The main experiments

```
#GCN
python train_rce.py --dataset Cora --encoder GCN --tau 0.1 --epochs 100
python train_rce.py --dataset CiteSeer --encoder GCN --cluster 4 --lr 0.05 --hidden 128 --tau 0.8
python train_rce.py --dataset PubMed --encoder GCN --tau 1.5
python train_rce.py --dataset DBLP --encoder GCN --weight_decay 1e-3 --hidden 128
python train_rce.py --dataset Facebook --encoder GCN --cluster 4 --lr 0.05 --weight_decay 0. --tau 1.

#SGC
python train_rce.py --dataset Cora --encoder SGC --tau 0.1 --epochs 100  --lr 0.01 --hidden 128
python train_rce.py --dataset CiteSeer --encoder SGC --cluster 4 --lr 0.02 --hidden 128 --tau 1.2
python train_rce.py --dataset PubMed --encoder SGC --tau 1.
python train_rce.py --dataset DBLP --encoder SGC --weight_decay 5e-4 --hidden 128
python train_rce.py --dataset Facebook --encoder SGC --cluster 4 --lr 0.05 --weight_decay 0. --tau 0.8

#MLP
python train_rce.py --dataset Cora --encoder MLP --tau 1.5
python train_rce.py --dataset CiteSeer --encoder MLP --cluster 4 --lr 0.01 --hidden 128 --tau 1.2
python train_rce.py --dataset PubMed --encoder MLP --lr 0.01 --hidden 128 --tau 1.5
python train_rce.py --dataset DBLP --encoder MLP --weight_decay 1e-3
python train_rce.py --dataset Facebook --encoder MLP --cluster 4 --tau 0.8

#GAT
python train_rce.py --dataset Cora --cluster 7 --hidden 16
python train_rce.py --dataset CiteSeer --cluster 4 --hidden 8 --lr 0.01 --wd 1e-3
python train_rce.py --dataset PubMed --heads 4 --hidden 8 --cluster 3 --lr 5e-3 --wd 1e-3 --att_drop2 0.
python train_rce.py --dataset DBLP --wd 5e-3
python train_rce.py --dataset Facebook --cluster 4 --heads 4 --lr 0.05 --wd 5e-6

###Graphsage
#ce
python train.py --dataset Cora
python train.py --dataset CiteSeer
python train.py --dataset PubMed
python train.py --dataset DBLP --wd 5e-3
python train.py --dataset Facebook --wd 0.
#rce
python train_rce.py --dataset Cora --cluster 10 --wd 5e-5 
python train_rce.py --dataset CiteSeer --lr 0.05 --hidden 128  
python train_rce.py --dataset PubMed 
python train_rce.py --dataset DBLP --lr 0.05 --wd 5e-3 
python train_rce.py --dataset Facebook --cluster 4 --wd 0.
```

## Citation
If you find our repository useful for your research, please consider citing our paper:
```
@article{miao2024rethinking,
  title={Rethinking Independent Cross-Entropy Loss For Graph-Structured Data},
  author={Miao, Rui and Zhou, Kaixiong and Wang, Yili and Liu, Ninghao and Wang, Ying and Wang, Xin},
  journal={arXiv preprint arXiv:2405.15564},
  year={2024}
}
```
