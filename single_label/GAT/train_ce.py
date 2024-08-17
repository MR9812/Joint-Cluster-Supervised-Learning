import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, CitationFull, FacebookPagePage
from torch_geometric.nn import GATConv

import random
import numpy as np
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborSampler

def set_seed():
    seed = 72
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask

def one_hot(x, class_count):
    res = torch.nn.functional.one_hot(x, class_count)
    res = res.float()
    return res
    
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, out_heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 8, out_channels, heads=out_heads, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    

def train(model, optimizer, data, labels, idx_train):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data, labels, idx_train, idx_val, idx_test):
    model.eval()
    
    y_pre = model(data.x, data.edge_index)
    acc_train = accuracy(y_pre[idx_train], labels[idx_train])
    acc_val = accuracy(y_pre[idx_val], labels[idx_val])
    acc_test = accuracy(y_pre[idx_test], labels[idx_test])
    loss = F.cross_entropy(y_pre[idx_val], labels[idx_val])
    
    return acc_train, acc_val, acc_test, loss
        
    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--device', type=int, default=0, help='GPU id.')
parser.add_argument('--hidden_channels', type=int, default=8)
parser.add_argument('--out_heads', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--att_drop1', type=float, default=0.6)
parser.add_argument('--att_drop2', type=float, default=0.6)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--patience', type=int, default=100)
args = parser.parse_args()
if args.device != -1 and torch.cuda.is_available():
    device = 'cuda:{}'.format(args.device)
else:
    device = 'cpu'
    
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)
if args.dataset=='DBLP':
    dataset = CitationFull(root=path, name= 'dblp', transform=T.NormalizeFeatures())
elif args.dataset == 'Facebook':
    dataset = FacebookPagePage(root=path, transform=T.NormalizeFeatures())
else:
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)


#show results
tests = []
set_seed()
for run in range(args.runs):
    if args.dataset != 'Cora' and args.dataset != 'CiteSeer' and args.dataset != 'PubMed':
        index_train = []
        index_val = []
        for i_label in range(data.y.max()+1):
            index_sub = [i for i,x in enumerate(data.y) if x==i_label]#train/val index
            index_sub = random.sample(index_sub, 50)
            index_train += index_sub[:20]
            index_val += index_sub[20:]
        
        index_train.sort()
        index_val.sort()
        index_train_val = index_val + index_train
        
        index_test = [i for i in range(data.y.shape[0]) if i not in index_train_val]
        
        train_mask = sample_mask(index_train, data.y.shape)#array([ True,  True,  True, ..., False, False, False])
        val_mask = sample_mask(index_val, data.y.shape)
        test_mask = sample_mask(index_test, data.y.shape)
        data.train_mask = torch.Tensor(train_mask).bool()
        data.val_mask = torch.Tensor(val_mask).bool()
        data.test_mask = torch.Tensor(test_mask).bool()
        
    model = GAT(dataset.num_features, args.hidden_channels, dataset.num_classes,
                args.out_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    
    best_val_acc = final_test_acc = 0
    best_val_loss = 999999999
    for epoch in range(1, args.epochs + 1):
        loss = train(model, optimizer, data, data.y, data.train_mask)
        train_acc, val_acc, test_acc, val_loss  = test(model, data, data.y, data.train_mask, data.val_mask, data.test_mask)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            final_test_acc = test_acc
            bad_counter = 0
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     final_test_acc = test_acc
        #     bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == args.patience:
            break
        print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}, '
              f'Final Test: {final_test_acc:.4f}')
    print(run+1, " result: ", final_test_acc)
    tests.append(final_test_acc.item())


print("=== Final ===")
print("test:", tests)
print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests):.6f}")
print(args)  
    
          
          
