## Joint-Cluster Supervised Learning Framework

This is the official implementation of the following paper:

> [Rethinking Independent Cross-Entropy Loss For Graph-Structured Data](https://arxiv.org/pdf/2405.15564)
> 
> Accepted by ICML 2024

In this work, we propose a new framework, termed joint-cluster supervised learning, to model the joint distribution of each node with its corresponding cluster. We learn the joint distribution of node and cluster labels conditioned on their representations, and train GNNs with the obtained joint loss. In this way, the data label reference signals extracted from the local cluster explicitly strengthen the discrimination ability on the target node. 

## Dependencies

- python 3.9.15
- torch    2.3.1+cu118
- torch_geometric 2.5.3

## The main experiments

```
single_label/run.sh
multi_label/run.sh
imbalance/run.sh
```
such as 
```
# GCN + joint-cluster loss
cd single_label/GCN_SGC_MLP
python train_rce.py --dataset Cora --encoder GCN --tau 0.1 --epochs 100
```

## Citation
If you find our repository useful for your research, please consider citing our paper:
```

@InProceedings{miao2024rethinking,
  title={Rethinking Independent Cross-Entropy Loss For Graph-Structured Data},
  author={Miao, Rui and Zhou, Kaixiong and Wang, Yili and Liu, Ninghao and Wang, Ying and Wang, Xin},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  pages={35570--35589},
  year={2024}
}
```
