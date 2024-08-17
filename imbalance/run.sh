# Imbalance

# LastFMAsia ce
python train_ce.py --dataset LastFMAsia --encoder GCN --lr 0.05 --weight_decay 0. --hidden 256 --epoch 500
python train_ce.py --dataset LastFMAsia --encoder SGC --lr 0.05 --weight_decay 0. --hidden 256 --epoch 500
python train_ce.py --dataset LastFMAsia --encoder MLP --lr 0.05 --weight_decay 0. --hidden 256 --epoch 500
# LastFMAsia rce
python train_rce.py --dataset LastFMAsia --encoder GCN --cluster 20 --lr 0.05 --weight_decay 0. --hidden 256 --epoch 500
python train_rce.py --dataset LastFMAsia --encoder SGC --cluster 20 --lr 0.05 --weight_decay 0. --hidden 256 --epoch 500
python train_rce.py --dataset LastFMAsia --encoder MLP --cluster 20 --lr 0.05 --weight_decay 0. --hidden 256 --epoch 500

# Arxiv ce
python arxiv-mlp-ce.py --encoder MLP 
python arxiv-mlp-ce.py --encoder SGC
python arxiv-train-ce.py --encoder GCN
# Arxiv rce
python arxiv-mlp-rce.py --encoder MLP
python arxiv-mlp-rce.py --encoder SGC
python arxiv-train-rce.py --encoder GCN