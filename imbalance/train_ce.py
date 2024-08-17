from __future__ import division
from __future__ import print_function
import random

import time
import argparse
import numpy as np
from copy import deepcopy as dcp
import scipy.sparse as sp
import math

import os.path as osp
import os
from torch_geometric.utils import to_dense_adj, add_self_loops, remove_self_loops
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborSampler

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from models import GCN, MLP
from torch_geometric.datasets import Planetoid, CitationFull, FacebookPagePage, LastFMAsia, WikipediaNetwork, Actor
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, f1_score
from torch_sparse import SparseTensor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def one_hot(x, class_count):
    res = torch.nn.functional.one_hot(x, class_count)
    res = res.float()
    return res

def cal_adj(edge_index, num_nodes, alpha=0.1, values=None):
    #import ipdb;ipdb.set_trace()
    #values = (alpha - values) / alpha
    #adj_tu = SparseTensor(row=edge_index[0], col=edge_index[1], value=values, sparse_sizes=(num_nodes, num_nodes))
    #import ipdb;ipdb.set_trace()
    adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
    adj_t = adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    DA = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj_t
    return DAD, DA

#data spilt
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask

#SGC 
def propagate(feature, A, order, alpha):
    y = feature
    for i in range(order):
        y = (1 - alpha) * (A @ y).detach_() + alpha * y
    return y.detach_()

def propagate_grand(feature, A, order):
    #feature = F.dropout(feature, args.dropout, training=training)
    x = feature
    y = feature
    for i in range(order):
        x = torch.spmm(A, x).detach_()
        #print(y.add_(x))
        y.add_(x)
        
    return y.div_(order+1.0).detach_()

#adj normalization
def adj_nor(edge):
    degree = torch.sum(edge, dim=1)
    degree = 1 / torch.sqrt(degree)
    degree = torch.diag(degree)
    adj = torch.mm(torch.mm(degree, edge), degree)
    return adj

#feature normalization
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train(model, optimizer, x, adj, y, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(x, adj)
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

def test(model, x, adj, y, train_mask, val_mask, test_mask, args):
    model.eval()
    pred = model(x, adj)
    aa = pred
    pred = pred.argmax(dim=-1)
    accs = []
    
    if args.metric == 'acc':
        for mask in [train_mask, val_mask, test_mask]:
            accs.append(int((pred[mask] == y[mask]).sum()) / int(mask.sum()))
    elif args.metric == 'macro':
        for mask in [train_mask, val_mask, test_mask]:
            accs.append(f1_score(y[mask].detach().cpu().numpy(), pred[mask].detach().cpu().numpy(), average='macro'))
    else:
        for mask in [train_mask, val_mask, test_mask]:
            accs.append(f1_score(y[mask].detach().cpu().numpy(), pred[mask].detach().cpu().numpy(), average='weighted'))
    
    accs.append(aa)
    return accs
    

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='LastFMAsia', help='Cora/CiteSeer/PubMed/')
parser.add_argument('--encoder', type=str, default='GCN', help='GCN/SGC/MLP/')
parser.add_argument('--runs', type=int, default=10,  help='run times.')
parser.add_argument('--device', type=int, default=0, help='GPU id.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--metric', type=str, default='acc', help='acc/macro/weight/')
args = parser.parse_args()

if args.device != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.device)
else:
    args.device = 'cpu'
device = args.device
path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset', args.dataset)
if args.dataset=='DBLP':
    dataset = CitationFull(root=path, name= 'dblp', transform=T.NormalizeFeatures())
elif args.dataset == 'Facebook':
    dataset = FacebookPagePage(root=path, transform=T.NormalizeFeatures())
elif args.dataset == 'LastFMAsia':
    dataset = LastFMAsia(root=path, transform=T.NormalizeFeatures())
elif args.dataset == 'cham':
    dataset = WikipediaNetwork(root=path, name= 'chameleon', geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
else:
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())

data = dataset[0]
#import ipdb;ipdb.set_trace()
#data process
x = data.x
x = normalize(x)
x = torch.from_numpy(x)
y = data.y
adj, _ = cal_adj(data.edge_index, data.num_nodes, alpha=0.1, values=None)

#SGC
if args.encoder == 'SGC':
    x = propagate(x, adj, 2, 0.)
    #x = propagate_grand(x, adj, 16)

x = x.to(device)
adj = adj.to(device)
y = y.to(device)
if args.dataset=='Cora' or args.dataset=='CiteSeer' or args.dataset=='PubMed':
    idx_train = data.train_mask.to(device)
    idx_val = data.val_mask.to(device)
    idx_test =  data.test_mask.to(device)

#main()
tests = []
set_seed(42)
for run in range(args.runs):
    if args.dataset != 'Cora' and args.dataset != 'CiteSeer' and args.dataset != 'PubMed':
        #other datasets data split
        index_train = []
        index_val = []
        for i_label in range(data.y.max()+1):
            index_sub = [i for i,x in enumerate(data.y) if x==i_label]#train/val index
            if args.dataset == 'LastFMAsia':
                index_sub = random.sample(index_sub,  int(len(index_sub)*0.5))
                index_train += index_sub[: int(len(index_sub)*0.5)]
                index_val += index_sub[int(len(index_sub)*0.5):]
            else:
                index_sub = random.sample(index_sub, 50)
                index_train += index_sub[:20]
                index_val += index_sub[20:]
        
        index_train.sort()
        index_val.sort()
        index_train_val = index_val + index_train
        index_test = [i for i in range(data.y.shape[0]) if i not in index_train_val]
        
        train_mask = sample_mask(index_train, data.y.shape)
        val_mask = sample_mask(index_val, data.y.shape)
        test_mask = sample_mask(index_test, data.y.shape)
        idx_train = torch.Tensor(train_mask).bool().to(device)
        idx_val = torch.Tensor(val_mask).bool().to(device)
        idx_test = torch.Tensor(test_mask).bool().to(device)
    
    # Model and optimizer
    if args.encoder == 'GCN':
        model = GCN(nfeat=x.shape[1],
                    nhid=args.hidden,
                    nclass=y.max().item() + 1,
                    dropout=args.dropout).to(device)
    else:
        model = MLP(nfeat=x.shape[1],
                    nhid=args.hidden,
                    nclass=y.max().item() + 1,
                    dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Train model
    best_val_acc = final_test_acc = 0
    t_total = time.time()
    total_train = 0
    total_test = 0
    for epoch in range(args.epochs):
        t_start1 = time.time()
        loss = train(model, optimizer, x, adj, y, idx_train)
        t_train = time.time()-t_start1
        t_start2 = time.time()
        train_acc, val_acc, test_acc, aa = test(model, x, adj, y, idx_train, idx_val, idx_test, args)
        t_test = time.time()-t_start2
        total_train = total_train + t_train
        total_test = total_test + t_test
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc
            bb = aa
        print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}, final_test_acc:{:4f}'.
                format(epoch, train_acc, val_acc, test_acc, final_test_acc))
    print("test acc: ", final_test_acc)
    print(total_train/200)
    print(total_test/200)
    #import ipdb;ipdb.set_trace()
    # list1 = [2,4,8,16,32,64]
    # for i in range(1, 10):
    #     for j in range(6):
    #         alpha = i/10
    #         K = list1[j]
    #         cc = bb
    #         for k in range(K):
    #             cc = (1 - alpha) * torch.matmul(adj, cc) + alpha * bb
    #         print(K, "#", alpha)
    #         final_test_acc = int((cc.argmax(-1)[idx_test] == y[idx_test]).sum()) / int(idx_test.sum())
    #         print(final_test_acc)
    tests.append(final_test_acc)


print("=== Final ===")
print("test:", tests)
print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests):.6f}")
print(args)

#后处理
#citeseer 16 0.1 71.02  hidden 128