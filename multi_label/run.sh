###SIGN
#python sign_ce.py --dataset yelp --hidden_channels 256
#python sign_ce.py --dataset amazon --hidden_channels 1024
#python sign_rce.py --dataset yelp --hidden_channels 512 --cluster 500
#python sign_rce.py --dataset amazon --hidden_channels 1024 --cluster 5000
###Graphsage
#python Graphsage_ce.py --dataset yelp
#python Graphsage_ce.py --dataset amazon --epoch 40 --eval_epo 20
#python Graphsage_rce.py --dataset yelp
#python Graphsage_rce.py --dataset amazon --epoch 40 --eval_epo 20
###Cluster_GCN
#python cluster_gcn_ce.py
#python cluster_gcn_ce.py --dataset amazon --num_partitions 6000 --eval_epo 19
#python cluster_gcn_rce.py
#python cluster_gcn_rce.py --dataset amazon --num_partitions 6000 --eval_epo 19