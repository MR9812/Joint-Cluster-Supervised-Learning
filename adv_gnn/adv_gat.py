import torch
import argparse
from deeprobust.graph.data import Dataset, Dpr2Pyg
from deeprobust.graph.defense import GAT
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PrePtbDataset
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='perturbation rate')
parser.add_argument('--runs', type=int, default=10, help='Runs.')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# use data splist provided by prognn
data = Dataset(root='/tmp/', name=args.dataset, setting='prognn')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

pyg_data = Dpr2Pyg(data)
if args.ptb_rate!= 0.0:
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

test_acc = torch.zeros(args.runs)
for run in range(args.runs):
    gat = GAT(nfeat=features.shape[1],
          nhid=8, heads=8,
          nclass=labels.max().item() + 1,
          dropout=0.5, device=device)
    gat = gat.to(device)
    
    gat.fit(pyg_data, verbose=True) # train with earlystopping
    test_acc[run] = gat.test()

print("=== Final ===")
print(torch.max(test_acc))
print(torch.min(test_acc))
print("10次平均",torch.mean(test_acc))
print("10次标准差",test_acc.std())
print(test_acc)



