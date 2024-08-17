## Joint-Cluster Supervised Learning Framework

This is the official implementation of the following paper:

> [Rethinking Independent Cross-Entropy Loss For Graph-Structured Data](https://arxiv.org/pdf/2405.15564)
> 
> Accepted by ICML 2024

In this work, we propose a new framework, termed joint-cluster supervised learning, to model the joint distribution of each node with its corresponding cluster. We learn the joint distribution of node and cluster labels conditioned on their representations, and train GNNs with the obtained joint loss. In this way, the data label reference signals extracted from the local cluster explicitly strengthen the discrimination ability on the target node. 

## The main experiments

```
single_label/run.sh
multi_label/run.sh
imbalance/run.sh
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
