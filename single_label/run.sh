# GCN ce
python train_ce.py --dataset Cora --encoder GCN
python train_ce.py --dataset CiteSeer --encoder GCN
python train_ce.py --dataset PubMed --encoder GCN
python train_ce.py --dataset DBLP --encoder GCN --weight_decay 5e-3
python train_ce.py --dataset Facebook --encoder GCN --lr 0.05 --weight_decay 0. --hidden 128
# GCN rce
python train_rce.py --dataset Cora --encoder GCN --tau 0.1 --epochs 100
python train_rce.py --dataset CiteSeer --encoder GCN --lr 0.05 --hidden 128 --tau 0.8 --device 3
python train_rce.py --dataset PubMed --encoder GCN --tau 1.5
python train_rce.py --dataset DBLP --encoder GCN --weight_decay 1e-3 --hidden 128
python train_rce.py --dataset Facebook --encoder GCN --cluster 4 --lr 0.05 --weight_decay 0. --tau 1.

# SGC ce
python train_ce.py --dataset Cora --encoder SGC
python train_ce.py --dataset CiteSeer --encoder SGC
python train_ce.py --dataset PubMed --encoder SGC
python train_ce.py --dataset DBLP --encoder SGC --weight_decay 5e-3 --lr 0.01
python train_ce.py --dataset Facebook --encoder SGC --lr 0.05 --weight_decay 0.
# SGC rce
python train_rce.py --dataset Cora --encoder SGC --tau 0.1 --epochs 100  --lr 0.01 --hidden 128
python train_rce.py --dataset CiteSeer --encoder SGC --cluster 4 --lr 0.05 --hidden 128 --tau 1.2
python train_rce.py --dataset PubMed --encoder SGC --tau 1.
python train_rce.py --dataset DBLP --encoder SGC --weight_decay 5e-4 --hidden 128
python train_rce.py --dataset Facebook --encoder SGC --cluster 4 --lr 0.05 --weight_decay 0. --tau 0.8

# MLP ce
python train_ce.py --dataset Cora --encoder MLP
python train_ce.py --dataset CiteSeer --encoder MLP
python train_ce.py --dataset PubMed --encoder MLP
python train_ce.py --dataset DBLP --encoder MLP --weight_decay 5e-3
python train_ce.py --dataset Facebook --encoder MLP
# MLP rce
python train_rce.py --dataset Cora --encoder MLP --tau 1.5
python train_rce.py --dataset CiteSeer --encoder MLP --cluster 4 --lr 0.01 --hidden 128 --tau 1.2
python train_rce.py --dataset PubMed --encoder MLP --lr 0.01 --hidden 128 --tau 1.5
python train_rce.py --dataset DBLP --encoder MLP --weight_decay 1e-3
python train_rce.py --dataset Facebook --encoder MLP --cluster 4 --tau 0.8

# GAT ce
python train_ce.py --dataset Cora
python train_ce.py --dataset CiteSeer
python train_ce.py --dataset PubMed --out_heads 8 --wd 1e-3
python train_ce.py --dataset DBLP --wd 5e-3
python train_ce.py --dataset Facebook --out_heads 4 --lr 0.01 --wd 5e-6
# GAT rce
python train_rce.py --dataset Cora --cluster 7 --hidden 16
python train_rce.py --dataset CiteSeer --cluster 4 --hidden 8 --lr 0.01 --wd 1e-3
python train_rce.py --dataset PubMed --out_heads 4 --hidden 8 --cluster 3 --lr 5e-3 --wd 1e-3 --att_drop2 0.
python train_rce.py --dataset DBLP --wd 5e-3
python train_rce.py --dataset Facebook --cluster 4 --out_heads 4 --lr 0.05 --wd 5e-6

# Graphsage ce
python train.py --dataset Cora
python train.py --dataset CiteSeer
python train.py --dataset PubMed
python train.py --dataset DBLP --wd 5e-3
python train.py --dataset Facebook --wd 0. --lr 0.05
# Graphsage rce
python train_rce.py --dataset Cora --cluster 10 --wd 5e-5 
python train_rce.py --dataset CiteSeer --lr 0.05 --hidden 256
python train_rce.py --dataset PubMed 
python train_rce.py --dataset DBLP --lr 0.05 --wd 5e-3 
python train_rce.py --dataset Facebook --cluster 4 --wd 0. --lr 0.05