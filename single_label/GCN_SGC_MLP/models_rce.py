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
        self.gc2 = GraphConvolution(nhid, nhid)
        
        #self.gc3 = GraphConvolution(nhid, nhid)
        #self.gc4 = GraphConvolution(nhid, nhid)
        #self.gc5 = GraphConvolution(nhid, nhid)
        #self.gc6 = GraphConvolution(nhid, nhid)
        #self.gc7 = GraphConvolution(nhid, nhid)
        #self.gc8 = GraphConvolution(nhid, nhid)
        
        self.fc1 = nn.Linear(nhid * 2, nclass * nclass)
        
        self.dropout = dropout
        self.nclass = nclass

    def forward(self, x, adj, cluster_id, cluster_index, is_val = False):
        x = F.relu(self.gc1(x, adj))
        
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.gc2(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.gc3(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.gc4(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.gc5(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.gc6(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.gc7(x, adj))
        #embeds = torch.mm(adj, x)
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(adj, x)
        
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
        #return x, embeds, cluster_features





class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()
        
        self.fc1 = torch.nn.Linear(nfeat, nhid)
        self.fc2 = torch.nn.Linear(2 * nhid, nclass * nclass)
        self.dropout = dropout
        self.nclass = nclass

    def forward(self, x, adj, cluster_id, cluster_index, is_val = False):
        #x = self.fc1(x)
        #embeds = x
        #x = F.relu(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        
        if is_val == False:
            cluster_features = torch.mm(cluster_id.t(), x[cluster_index]) / cluster_id.sum(0).unsqueeze(1)
            x1 = cluster_features[cluster_id.argmax(1)]
            x = torch.cat((torch.cat((x[cluster_index], x1), 1), torch.cat((x1, x[cluster_index]), 1)), 0)
        else:
            cluster_features = torch.mm(cluster_id[cluster_index].t(), x[cluster_index]) / cluster_id[cluster_index].sum(0).unsqueeze(1)
            x1 = cluster_features[cluster_id.argmax(1)]
            x = torch.cat((x, x1), 1)
        
        x = self.fc2(x)

        return x
        #return x, embeds, cluster_features


    
                                            