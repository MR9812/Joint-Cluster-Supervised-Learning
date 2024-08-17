import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import math
from torch_geometric.nn import GATConv, GCNConv, SGConv
from torch_geometric.nn.inits import glorot, zeros


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, is_aug=False, aug_rate = 0.):
        if is_aug == True:
            x = F.dropout(x, aug_rate, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    
    
class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()
        
        self.fc1 = torch.nn.Linear(nfeat, nhid)
        self.fc2 = torch.nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, is_aug=False, aug_rate = 0.):
        if is_aug == True:
            x = F.dropout(x, aug_rate, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        return x

#drop 0.7, 80, 0.8, 0.7 citeseer