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

from models_rce import GCN, MLP
from torch_geometric.datasets import Planetoid, CitationFull, FacebookPagePage, LastFMAsia, WikiCS, Coauthor, Amazon, Actor
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, f1_score


def one_hot(x, class_count):
    return torch.eye(class_count)[x,:]

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
        y = (1 - alpha) * torch.spmm(A, y).detach_() + alpha * y
    return y.detach_()

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

def train(model, optimizer, epoch, features, adj, idx_train, idx_val, idx_test, labels, index, num_classes, args, cluster_train_id, cluster_train_label, cluster_val_id, cluster_val_label):
    global best_model
    global best_val_acc
    t = time.time()
    model.train()
    optimizer.zero_grad()
    
    y_pre = model(features, adj, cluster_train_id, idx_train)
    y_pre = F.softmax(y_pre / args.tau, dim=1)
    
    #acc_train
    p_label = y_pre.view(-1, num_classes, num_classes)
    p_label = p_label.sum(dim=2)
    acc_train = accuracy(p_label[:int(p_label.shape[0]/2)], labels[idx_train])
    #acc_train = f1_score(labels[idx_train].detach().cpu().numpy(), p_label[:int(p_label.shape[0]/2)].argmax(dim=-1, keepdim=True).detach().cpu().numpy(), average='micro')
    #acc_train = f1_score(labels[idx_train].detach().cpu().numpy(), p_label[:int(p_label.shape[0]/2)].argmax(dim=-1, keepdim=True).detach().cpu().numpy(), average='macro')
    #acc_train = f1_score(labels[idx_train].detach().cpu().numpy(), p_label[:int(p_label.shape[0]/2)].argmax(dim=-1, keepdim=True).detach().cpu().numpy(), average='weighted')
    
    #relation ce : ground truth
    p_label1 = one_hot(labels[idx_train],  num_classes).cuda()
    p_label2 = cluster_train_label[cluster_train_id.argmax(1)].cuda()
    rce_label = torch.cat((torch.matmul(p_label1.unsqueeze(2), p_label2.unsqueeze(1)), torch.matmul(p_label2.unsqueeze(2), p_label1.unsqueeze(1))), 0)
    rce_label = rce_label.view(-1, num_classes * num_classes)
    
    #rce_pre
    rce_pre = torch.log(y_pre + 1e-8)
    loss = -torch.sum(rce_pre * rce_label) / rce_pre.shape[0]
    loss.backward()
    optimizer.step()
            
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        y_pre = model(features, adj, one_hot(index, args.cluster).cuda(), idx_train+idx_val, True)
    
    #acc_val
    y_pre = F.softmax(y_pre, dim=1)
    p_label = y_pre.view(-1, num_classes, num_classes)
    p_label = p_label.sum(dim=2)
    acc_val = accuracy(p_label[idx_val], labels[idx_val])   #gai
    #acc_val = f1_score(labels[idx_val].detach().cpu().numpy(), p_label[idx_val].argmax(dim=-1, keepdim=True).detach().cpu().numpy(), average='micro')
    #acc_val = f1_score(labels[idx_val].detach().cpu().numpy(), p_label[idx_val].argmax(dim=-1, keepdim=True).detach().cpu().numpy(), average='macro')
    #acc_val = f1_score(labels[idx_val].detach().cpu().numpy(), p_label[idx_val].argmax(dim=-1, keepdim=True).detach().cpu().numpy(), average='weighted')
    
    acc_test = accuracy(p_label[idx_test], labels[idx_test])
    #acc_test = f1_score(labels[idx_test].detach().cpu().numpy(), p_label[idx_test].argmax(dim=-1, keepdim=True).detach().cpu().numpy(), average='micro')
    #acc_test = f1_score(labels[idx_test].detach().cpu().numpy(), p_label[idx_test].argmax(dim=-1, keepdim=True).detach().cpu().numpy(), average='macro')
    #acc_test = f1_score(labels[idx_test].detach().cpu().numpy(), p_label[idx_test].argmax(dim=-1, keepdim=True).detach().cpu().numpy(), average='weighted')
    
    #relation ce : ground truth
    p_label1 = one_hot(labels[idx_val],  num_classes).cuda()
    p_label2 = cluster_val_label[index[idx_val]].cuda()
    rce_label = torch.matmul(p_label1.unsqueeze(2), p_label2.unsqueeze(1))
    rce_label = rce_label.view(-1, num_classes * num_classes)
    
    #rce_pre
    rce_pre = torch.log(y_pre[idx_val] + 1e-8)
    loss = -torch.sum(rce_pre * rce_label) / rce_pre.shape[0]
    
    if acc_val > best_val_acc:
        best_val_acc = acc_val
        best_model = dcp(model)
            
    print('Epoch: {:04d}'.format(epoch+1),
          'loss: {:.4f}'.format(loss.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'acc_test: {:.4f}'.format(acc_test.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(model, features, adj, labels, idx_train, idx_val, idx_test, index, num_classes, args, cluster_val_id, cluster_val_label):
    model.eval()
    y_pre = model(features, adj, one_hot(index, args.cluster).cuda(), idx_train+idx_val, True)
    
    y_pre = F.softmax(y_pre, dim=1)
    p_label = y_pre.view(-1, num_classes, num_classes)
    p_label = p_label.sum(dim=2)
    acc_test = accuracy(p_label[idx_test], labels[idx_test])
    #acc_test = f1_score(labels[idx_test].detach().cpu().numpy(), p_label[idx_test].argmax(dim=-1, keepdim=True).detach().cpu().numpy(), average='micro')
    #acc_test = f1_score(labels[idx_test].detach().cpu().numpy(), p_label[idx_test].argmax(dim=-1, keepdim=True).detach().cpu().numpy(), average='macro')
    #acc_test = f1_score(labels[idx_test].detach().cpu().numpy(), p_label[idx_test].argmax(dim=-1, keepdim=True).detach().cpu().numpy(), average='weighted')

    print("Test set results:",
          #"loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test
    

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--dataset', type=str, default='Cora', help='Cora/CiteSeer/PubMed/')
parser.add_argument('--encoder', type=str, default='GCN', help='GCN/SGC/MLP/')
parser.add_argument('--seed', type=int, default=42,  help='Random seed.')
parser.add_argument('--runs', type=int, default=10,  help='run times.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--cluster', type=int, default=5, help='Number of clusters.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--tau', type=float, default=1.0, help='softmax tempurate.')
parser.add_argument('--metric', type=str, default='acc', help='acc/macro/weight/')

args = parser.parse_args()
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)

if args.dataset=='DBLP':
    dataset = CitationFull(root=path, name= 'dblp', transform=T.NormalizeFeatures())
elif args.dataset == 'Facebook':
    dataset = FacebookPagePage(root=path, transform=T.NormalizeFeatures())
elif args.dataset == 'LastFMAsia':
    dataset = LastFMAsia(root=path, transform=T.NormalizeFeatures())
else:
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())

data = dataset[0]
#import ipdb;ipdb.set_trace()
#cluster split
cluster_data = ClusterData(data, num_parts=args.cluster, recursive=False)
index = torch.zeros(data.x.shape[0])
for i in range(args.cluster):
    index[cluster_data.perm[cluster_data.partptr[i]:cluster_data.partptr[i+1]]] = i
index = index.long().cuda()

#data process
features = data.x
features = normalize(features)
features = torch.from_numpy(features)
labels = data.y
adj = to_dense_adj(add_self_loops(remove_self_loops(data.edge_index)[0])[0])[0]
adj = adj_nor(adj)

#SGC
if args.encoder == 'SGC':
    features = propagate(features, adj, 2, 0.)

features = features.cuda()
adj = adj.cuda()
labels = labels.cuda()
if args.dataset=='Cora' or args.dataset=='CiteSeer' or args.dataset=='PubMed':
    idx_train = data.train_mask.cuda()
    idx_val = data.val_mask.cuda()
    idx_test =  data.test_mask.cuda()

#main()
test_acc = torch.zeros(args.runs)
test_acc = test_acc.cuda()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

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
        idx_train = torch.Tensor(train_mask).bool().cuda()
        idx_val = torch.Tensor(val_mask).bool().cuda()
        idx_test = torch.Tensor(test_mask).bool().cuda()
    
    cluster_train_id = one_hot(index[idx_train], args.cluster).cuda()
    cluster_train_label = one_hot(labels[idx_train], dataset.num_classes).cuda()
    cluster_train_label = torch.mm(cluster_train_id.t(), cluster_train_label)
    cluster_train_label = cluster_train_label / cluster_train_label.sum(1).unsqueeze(1)
    
    cluster_val_id = one_hot(index[idx_train+idx_val], args.cluster).cuda()
    cluster_val_label = one_hot(labels[idx_train+idx_val], dataset.num_classes).cuda()
    cluster_val_label = torch.mm(cluster_val_id.t(), cluster_val_label) 
    cluster_val_label = cluster_val_label / cluster_val_label.sum(1).unsqueeze(1)
    
    
    best_model = None
    best_val_acc = 0.0
    # Model and optimizer
    if args.encoder == 'GCN':
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout).cuda()
    else:
        model = MLP(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout).cuda()
    optimizer = optim.Adam(model.parameters(),
                lr=args.lr, weight_decay=args.weight_decay)
    
    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(model, optimizer, epoch, features, adj, idx_train, idx_val, idx_test, labels, index, dataset.num_classes, args, cluster_train_id, cluster_train_label, cluster_val_id, cluster_val_label)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Testing
    test_acc[run] = test(best_model, features, adj, labels, idx_train, idx_val, idx_test, index, dataset.num_classes, args, cluster_val_id, cluster_val_label)


print("=== Final ===")
print(torch.max(test_acc))
print(torch.min(test_acc))
print("10次平均",torch.mean(test_acc))
print("10次标准差",test_acc.std())
print(test_acc)



