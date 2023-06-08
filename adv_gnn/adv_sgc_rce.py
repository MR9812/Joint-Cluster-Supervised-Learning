import torch
import argparse
from deeprobust.graph.data import Dataset, Dpr2Pyg
from model_sgc_rce import SGC
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PrePtbDataset
import random
import numpy as np
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborSampler

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='perturbation rate')
parser.add_argument('--runs', type=int, default=10, help='Runs.')
parser.add_argument('--cluster', type=int, default=5, help='Runs.')
parser.add_argument('--epoch', type=int, default=200, help='epochs.')
parser.add_argument('--hidden', type=int, default=64, help='hidden.')
parser.add_argument('--num_layers', type=int, default=3, help='num layers.')
parser.add_argument('--lr', type=float, default=0.1,  help='learning rate')
parser.add_argument('--wd', type=float, default=5e-4,  help='weight_decay')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# use data splist provided by prognn
data = Dataset(root='/tmp/', name=args.dataset, setting='prognn')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

pyg_data = Dpr2Pyg(data)
if args.ptb_rate != 0.0:
    # load pre-attacked graph by Zugner: https://github.com/danielzuegner/gnn-meta-attack
    print('==================')
    print('=== load graph perturbed by Zugner metattack (under prognn splits) ===')
    perturbed_data = PrePtbDataset(root='/tmp/',
            name=args.dataset,
            attack_method='meta',
            ptb_rate=args.ptb_rate)
    perturbed_adj = perturbed_data.adj
    pyg_data.update_edge_index(perturbed_adj) # inplace operation
    
#main
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

cluster_data = ClusterData(pyg_data[0], num_parts=args.cluster, recursive=False)
index = torch.zeros(pyg_data[0].x.shape[0])
for i in range(args.cluster):
    index[cluster_data.perm[cluster_data.partptr[i]:cluster_data.partptr[i+1]]] = i
index = index.long().cuda()

test_acc = torch.zeros(args.runs)
for run in range(args.runs):
    sgc = SGC(nfeat=features.shape[1],
          nclass=labels.max().item() + 1,
          cluster = args.cluster,
          nhid = args.hidden,
          lr=args.lr, 
          K = args.num_layers,
          weight_decay = args.wd,
          device=device)
    sgc = sgc.to(device)
    sgc.fit(pyg_data, index, args.epoch, verbose=True) # train with earlystopping
    test_acc[run] = sgc.test(index)

print("=== Final ===")
print(torch.max(test_acc))
print(torch.min(test_acc))
print("10 mean",torch.mean(test_acc))
print("10 std",test_acc.std())
print(test_acc)



