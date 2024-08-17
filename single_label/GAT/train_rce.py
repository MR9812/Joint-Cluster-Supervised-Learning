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
#from Data_air import Airports

#cluster, 3, 4,5,7,10,20
#hidden,  8,16,32
#heads,   1,2,4,8,16
#lr,      5e-3,1e-2,1e-3,5e-2
#wd,      5e-4,1e-3,5e-3,1e-4
#dropout  0.3,0.4,0.5,0.6,0.7 4个dropout


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask

def one_hot(x, class_count):
    res = torch.nn.functional.one_hot(x, class_count)
    res = res.float()
    return res

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout, att_drop1, att_drop2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, 8, dropout=att_drop1)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(2 * hidden_channels * 8, out_channels * out_channels, heads=heads, concat=False, dropout=att_drop2)
        #self.fc1 = nn.Linear(2 * hidden_channels * 8, out_channels * out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, cluster_id, cluster_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        #x = F.dropout(x, p=self.dropout, training=self.training)
        #x = F.elu(self.conv2(x, edge_index))
        #x = F.dropout(x, p=self.dropout, training=self.training)
        
        cluster_features = torch.mm(cluster_id[cluster_index].t(), x[cluster_index]) / cluster_id[cluster_index].sum(0).unsqueeze(1)
        x1 = cluster_features[cluster_id.argmax(1)]
        x = torch.cat((torch.cat((x, x1), 1), torch.cat((x1, x), 1)), 0)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, torch.cat((edge_index, edge_index+data.x.shape[0]), 1))
        #x = self.fc1(x)
        
        return x
    

def train(model, optimizer, data, num_classes, args, index, cluster_train_id, cluster_train_label, cluster_val_id, cluster_val_label):
    model.train()
    optimizer.zero_grad()
    
    y_pre = model(data.x, data.edge_index, one_hot(index, args.cluster).to(args.device), data.train_mask)
    y_pre = F.softmax(y_pre/args.tau, dim=1)
    
    #relation ce : ground truth
    labels = data.y
    p_label1 = one_hot(labels[data.train_mask],  num_classes).to(args.device)
    p_label2 = cluster_train_label[cluster_train_id.argmax(1)].to(args.device)
    
    #rce_label = torch.cat((torch.matmul(p_label1.unsqueeze(2), p_label2.unsqueeze(1)), torch.matmul(p_label2.unsqueeze(2), p_label1.unsqueeze(1))), 0)
    #rce_label = rce_label.view(-1, num_classes * num_classes)
    #rce_pre
    #rce_pre = torch.log(y_pre + 1e-8)
    #rce_pre = torch.cat((rce_pre[:int(y_pre.shape[0]/2)][data.train_mask], rce_pre[int(y_pre.shape[0]/2):][data.train_mask]), 0)
    #loss = -torch.sum(rce_pre * rce_label) / rce_pre.shape[0]
    
    rce_label1 = torch.matmul(p_label1.unsqueeze(2), p_label2.unsqueeze(1))
    rce_label1 = rce_label1.view(-1, num_classes * num_classes)
    rce_label2 = torch.matmul(p_label2.unsqueeze(2), p_label1.unsqueeze(1))
    rce_label2 = rce_label2.view(-1, num_classes * num_classes)
    #rce_pre
    rce_pre = torch.log(y_pre + 1e-8)
    loss_train1 = -torch.sum(rce_pre[:int(y_pre.shape[0]/2)][data.train_mask] * rce_label1) / data.train_mask.sum()
    loss_train2 = -torch.sum(rce_pre[int(y_pre.shape[0]/2):][data.train_mask] * rce_label2) / data.train_mask.sum()
    loss = loss_train1 + loss_train2
    
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data, num_classes, args, index, cluster_train_id, cluster_train_label, cluster_val_id, cluster_val_label):
    model.eval()
    
    y_pre = model(data.x, data.edge_index, one_hot(index, args.cluster).to(args.device), data.train_mask+data.val_mask)
    y_pre = F.softmax(y_pre/1, dim=1)
    
    #acc
    y_pred = y_pre[0: int(y_pre.shape[0]/2)].view(-1, num_classes, num_classes)
    y_pred = y_pred.sum(dim=2)
    y_pred = y_pred.argmax(dim=-1, keepdim=True).squeeze(1)
    
    #val loss
    labels = data.y
    p_label1 = one_hot(labels[data.val_mask], num_classes).to(args.device)
    p_label2 = cluster_val_label[index[data.val_mask]].to(args.device)
    rce_label = torch.cat((torch.matmul(p_label1.unsqueeze(2), p_label2.unsqueeze(1)), torch.matmul(p_label2.unsqueeze(2), p_label1.unsqueeze(1))), 0)
    rce_label = rce_label.view(-1, num_classes * num_classes)
    
    #rce_pre
    rce_pre = torch.log(y_pre + 1e-8)
    rce_pre = torch.cat((rce_pre[:int(y_pre.shape[0]/2)][data.val_mask], rce_pre[int(y_pre.shape[0]/2):][data.val_mask]), 0)
    loss = -torch.sum(rce_pre * rce_label) / rce_pre.shape[0]
    
    #acc 
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((y_pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs[0], accs[1], accs[2], loss
        
    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--device', type=int, default=0, help='GPU id.')
parser.add_argument('--hidden_channels', type=int, default=8)
parser.add_argument('--out_heads', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--att_drop1', type=float, default=0.6)
parser.add_argument('--att_drop2', type=float, default=0.6)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--seed', type=int, default=72)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--cluster', type=int, default=5, help='Cluster number.')
parser.add_argument('--tau', type=float, default=1.0, help='tau.')
args = parser.parse_args()
if args.device != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.device)
else:
    args.device = 'cpu'
device = args.device
    
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)
if args.dataset=='DBLP':
    dataset = CitationFull(root=path, name= 'dblp', transform=T.NormalizeFeatures())
elif args.dataset == 'Facebook':
    dataset = FacebookPagePage(root=path, transform=T.NormalizeFeatures())
else:
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)

#cluster split
cluster_data = ClusterData(data, num_parts=args.cluster, recursive=False)
index = torch.zeros(data.x.shape[0])
for i in range(args.cluster):
    index[cluster_data.partition.node_perm[cluster_data.partition.partptr[i]:cluster_data.partition.partptr[i+1]]] = i
index = index.long().to(device)


#show results
test_res = torch.zeros(args.runs)
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

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
    
    labels = data.y
    cluster_train_id = one_hot(index[data.train_mask], args.cluster).to(device)
    cluster_train_label = one_hot(labels[data.train_mask], dataset.num_classes).to(device)
    cluster_train_label = torch.mm(cluster_train_id.t(), cluster_train_label)
    cluster_train_label = cluster_train_label / cluster_train_label.sum(1).unsqueeze(1)
    
    cluster_val_id = one_hot(index[data.train_mask+data.val_mask ], args.cluster).to(device)
    cluster_val_label = one_hot(labels[data.train_mask+data.val_mask ], dataset.num_classes).to(device)
    cluster_val_label = torch.mm(cluster_val_id.t(), cluster_val_label) 
    cluster_val_label = cluster_val_label / cluster_val_label.sum(1).unsqueeze(1)
        
    model = GAT(dataset.num_features, args.hidden_channels, dataset.num_classes,
                args.out_heads, args.dropout, args.att_drop1, args.att_drop2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    
    best_test_acc = 0
    best_val_loss = 999999999
    for epoch in range(1, args.epochs + 1):
        loss = train(model, optimizer, data, dataset.num_classes, args, index, cluster_train_id, cluster_train_label, cluster_val_id, cluster_val_label)
        train_acc, val_acc, test_acc, val_loss  = test(model, data, dataset.num_classes, args, index, cluster_train_id, cluster_train_label, cluster_val_id, cluster_val_label)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_acc = test_acc
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == args.patience:
            break
        print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}, '
              f'Final Test: {best_test_acc:.4f}')
    print(" ", run+1, " result: ", best_test_acc)
    test_res[run] = best_test_acc
 
print("=== Final ===")
print(torch.max(test_res))
print(torch.min(test_res))
print("mean",torch.mean(test_res))
print("std",test_res.std())
print(test_res)  
    
          
          
