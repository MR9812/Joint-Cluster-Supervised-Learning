import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PtbDataset, PrePtbDataset
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--runs', type=int, default=10, help='Runs.')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# We can just use setting='prognn' to get the splits
data = Dataset(root='/tmp/', name=args.dataset, setting='prognn')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

if args.ptb_rate == 0.0:
    perturbed_adj = adj
else:
    # load pre-attacked graph by Zugner: https://github.com/danielzuegner/gnn-meta-attack
    print('==================')
    print('=== load graph perturbed by Zugner metattack (under prognn splits) ===')
    perturbed_data = PrePtbDataset(root='/tmp/',
            name=args.dataset,
            attack_method='meta',
            ptb_rate=args.ptb_rate)
    perturbed_adj = perturbed_data.adj


#main
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

test_acc = torch.zeros(args.runs)
for run in range(args.runs):
    model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
    model = model.to(device)
    
    model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
    model.eval()
    test_acc[run] = model.test(idx_test)
    
print("=== Final ===")
print(torch.max(test_acc))
print(torch.min(test_acc))
print("10次平均",torch.mean(test_acc))
print("10次标准差",test_acc.std())
print(test_acc)



