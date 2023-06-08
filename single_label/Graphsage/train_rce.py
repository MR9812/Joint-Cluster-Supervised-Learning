import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, FacebookPagePage, LastFMAsia
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import argparse
import random
import numpy as np
import torch.optim as optim
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborSampler
import torch.nn as nn
from Data_air import Airports, AttributedGraphDataset
from sklearn.metrics import roc_auc_score, f1_score


def one_hot(x, class_count):
    res = torch.eye(class_count).cuda()
    return res[x,:]
    
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.fc1 = nn.Linear(hidden_channels * 2, out_channels * out_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.fc1.reset_parameters()

    def forward(self, x, adj_t, cluster_id, cluster_index, is_val = False):
        for i, conv in enumerate(self.convs[:]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        if is_val == False:
            cluster_features = torch.mm(cluster_id.t(), x[cluster_index]) / cluster_id.sum(0).unsqueeze(1)
            x1 = cluster_features[cluster_id.argmax(1)]
            x = torch.cat((torch.cat((x[cluster_index], x1), 1), torch.cat((x1, x[cluster_index]), 1)), 0)
        else:
            cluster_features = torch.mm(cluster_id[cluster_index].t(), x[cluster_index]) / cluster_id[cluster_index].sum(0).unsqueeze(1)
            x1 = cluster_features[cluster_id.argmax(1)]
            x = torch.cat((x, x1), 1)
        
        x = self.fc1(x)
        return x


def train(model, optimizer, data, index, args):
    model.train()
    optimizer.zero_grad()
    
    labels = data.y
    train_idx = data.train_mask
    cluster_train_id = one_hot(index[train_idx], args.cluster).cuda()
    cluster_train_label = one_hot(labels[train_idx], args.num_classes).cuda()
    cluster_train_label = torch.mm(cluster_train_id.t(), cluster_train_label)
    cluster_train_label = cluster_train_label / cluster_train_label.sum(1).unsqueeze(1)
    
    y_pre = model(data.x, data.adj_t, cluster_train_id, train_idx)
    y_pre = F.softmax(y_pre / args.tau, dim=1)
    
    #relation ce : ground truth
    p_label1 = one_hot(labels[train_idx],  args.num_classes).cuda()
    p_label2 = cluster_train_label[cluster_train_id.argmax(1)].cuda()
    rce_label = torch.cat((torch.matmul(p_label1.unsqueeze(2), p_label2.unsqueeze(1)), torch.matmul(p_label2.unsqueeze(2), p_label1.unsqueeze(1))), 0)
    rce_label = rce_label.view(-1, args.num_classes * args.num_classes)
    
    #rce_pre
    rce_pre = torch.log(y_pre + 1e-8)
    loss = -torch.sum(rce_pre * rce_label) / rce_pre.shape[0]
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data, index, args):
    model.eval()
    labels = data.y
    cluster_val_id = one_hot(index[data.train_mask+data.val_mask], args.cluster).cuda()
    cluster_val_label = one_hot(labels[data.train_mask+data.val_mask], args.num_classes).cuda()
    cluster_val_label = torch.mm(cluster_val_id.t(), cluster_val_label) 
    cluster_val_label = cluster_val_label / cluster_val_label.sum(1).unsqueeze(1)
    
    y_pre = model(data.x, data.adj_t, one_hot(index, args.cluster).cuda(), data.train_mask+data.val_mask, True)
    y_pre = F.softmax(y_pre, dim=1)
    y_pred = y_pre.view(-1, args.num_classes, args.num_classes)
    y_pred = y_pred.sum(dim=2)
    y_pred = y_pred.argmax(dim=-1, keepdim=True).squeeze()
    
    idx_train = data.train_mask
    idx_val = data.val_mask
    idx_test = data.test_mask
    labels = data.y
    
    acc_train = int((y_pred[idx_train] == labels[idx_train]).sum()) / int(idx_train.sum())
    acc_val = int((y_pred[idx_val] == labels[idx_val]).sum()) / int(idx_val.sum())
    acc_test = int((y_pred[idx_test] == labels[idx_test]).sum()) / int(idx_test.sum())
    
    #relation ce : ground truth
    p_label1 = one_hot(labels[data.val_mask],  args.num_classes).cuda()
    p_label2 = cluster_val_label[index[data.val_mask]].cuda()
    rce_label = torch.matmul(p_label1.unsqueeze(2), p_label2.unsqueeze(1))
    rce_label = rce_label.view(-1, args.num_classes * args.num_classes)
    
    #rce_pre
    rce_pre = torch.log(y_pre[data.val_mask] + 1e-8)
    loss = -torch.sum(rce_pre * rce_label) / rce_pre.shape[0]
    
    return acc_train, acc_val, acc_test, loss

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset', type=str, default='Cora', help='Cora/CiteSeer/PubMed/')
parser.add_argument('--runs', type=int, default=10, help='Runs')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--num_layers', type=int, default=2, help='Number of GCN layers.')
parser.add_argument('--cluster', type=int, default=5, help='Number of Cluster.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--tau', type=float, default=1.0, help='softmax tempurate.')
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)
transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
if args.dataset=='DBLP':
    dataset = CitationFull(root=path, name= 'dblp', transform=transform)
elif args.dataset == 'Facebook':
    dataset = FacebookPagePage(root=path, transform=transform)
elif args.dataset == 'LastFMAsia':
    dataset = LastFMAsia(root=path, transform=transform)
else:
    dataset = Planetoid(path, args.dataset, transform=transform)

args.num_classes = dataset.num_classes
data = dataset[0]
data.adj_t = gcn_norm(data.adj_t)  # Pre-process GCN normalization.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

if args.dataset=='DBLP':
    dataset1 = CitationFull(root=path, name= 'dblp')
elif args.dataset == 'Facebook':
    dataset1 = FacebookPagePage(root=path)
elif args.dataset == 'LastFMAsia':
    dataset1 = LastFMAsia(root=path)
else:
    dataset1 = Planetoid(path, args.dataset)
data1 = dataset1[0]
cluster_data = ClusterData(data1, num_parts=args.cluster, recursive=False)
index = torch.zeros(data.x.shape[0])
for i in range(args.cluster):
    index[cluster_data.perm[cluster_data.partptr[i]:cluster_data.partptr[i+1]]] = i
index = index.long().cuda()

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
        data.train_mask = torch.Tensor(train_mask).bool().cuda()
        data.val_mask = torch.Tensor(val_mask).bool().cuda()
        data.test_mask = torch.Tensor(test_mask).bool().cuda()
        
    model = SAGE(in_channels=data.x.shape[1], hidden_channels=args.hidden, out_channels=args.num_classes, 
                 num_layers=args.num_layers, dropout=args.dropout).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    best_test_acc = 0
    best_val_acc = 0
    best_val_loss = 99999999
    for epoch in range(1, args.epochs + 1):
        loss = train(model, optimizer, data, index, args)
        train_acc, val_acc, test_acc, val_loss = test(model, data, index, args)
        if val_acc > best_val_acc :
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_test_acc = test_acc
        print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}, '
              f'Final Test: {best_test_acc:.4f}')
    print(run+1, " final test:", best_test_acc)
    test_res[run] = best_test_acc
 
print("=== Final ===")
print(torch.max(test_res))
print(torch.min(test_res))
print("10 mean",torch.mean(test_res))
print("10 std",test_res.std())
print(test_res)  
    
    
    