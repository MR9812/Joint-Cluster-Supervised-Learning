import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from arxiv_logger import Logger

import torch.nn as nn
import random
import numpy as np
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborSampler
from sklearn.metrics import roc_auc_score, f1_score


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        #self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        self.fc1 = nn.Linear(hidden_channels * 2, out_channels * out_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.fc1.reset_parameters()

    def forward(self, x, adj_t, cluster_id, cluster_index, is_val = False):
        for i, conv in enumerate(self.convs[:]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
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
        
        #x = self.bn(x)
        x = self.fc1(x)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        #self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.fc1 = nn.Linear(hidden_channels * 2, out_channels * out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.fc1.reset_parameters()

    def forward(self, x, adj_t, cluster_id, cluster_index, is_val = False):
        for i, conv in enumerate(self.convs[:]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
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
        
        #x = self.bn(x)
        x = self.fc1(x)
        
        return x

def one_hot(x, class_count):
    res = torch.eye(class_count).cuda()
    return res[x,:]

def train(model, args, data, index, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    
    labels = data.y.squeeze(1)
    #import ipdb;ipdb.set_trace()
    cluster_train_id = one_hot(index[train_idx], args.cluster).cuda()
    cluster_train_label = one_hot(labels[train_idx], args.num_classes).cuda()
    cluster_train_label = torch.mm(cluster_train_id.t(), cluster_train_label)
    cluster_train_label = cluster_train_label / cluster_train_label.sum(1).unsqueeze(1)
    
    y_pre = model(data.x, data.adj_t, cluster_train_id, train_idx)
    y_pre = F.softmax(y_pre / args.tau, dim=1) #>1
    
    #acc_train
    p_label = y_pre[:int(y_pre.shape[0]/2)].view(-1, args.num_classes, args.num_classes)
    p_label = p_label.sum(dim=2)
    #acc_train = accuracy(p_label, labels[idx_train])
    
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

    return loss.item()


@torch.no_grad()
def test(model, args, data, index, train_idx, valid_idx, split_idx, evaluator):
    model.eval()
    
    labels = data.y.squeeze(1)
    cluster_val_id = one_hot(index[torch.cat((train_idx, valid_idx),0)], args.cluster).cuda()
    cluster_val_label = one_hot(labels[torch.cat((train_idx, valid_idx),0)], args.num_classes).cuda()
    cluster_val_label = torch.mm(cluster_val_id.t(), cluster_val_label) 
    cluster_val_label = cluster_val_label / cluster_val_label.sum(1).unsqueeze(1)
    
    y_pre = model(data.x, data.adj_t, one_hot(index, args.cluster).cuda(), torch.cat((train_idx, valid_idx),0), True)
    y_pre = F.softmax(y_pre, dim=1)
    y_pre = y_pre.view(-1, args.num_classes, args.num_classes)
    y_pre = y_pre.sum(dim=2)
    y_pred = y_pre.argmax(dim=-1, keepdim=True)
    
    if args.metric == 'acc':
        train_acc = evaluator.eval({
            'y_true': data.y[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': data.y[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': data.y[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
        })['acc']
    else:
        train_acc = f1_score(labels[split_idx['train']].detach().cpu().numpy(), y_pred[split_idx['train']].detach().cpu().numpy(), average='weighted')
        valid_acc = f1_score(labels[split_idx['valid']].detach().cpu().numpy(), y_pred[split_idx['valid']].detach().cpu().numpy(), average='weighted')
        test_acc = f1_score(labels[split_idx['test']].detach().cpu().numpy(), y_pred[split_idx['test']].detach().cpu().numpy(), average='weighted')
        #train_acc = f1_score(labels[split_idx['train']].detach().cpu().numpy(), y_pred[split_idx['train']].detach().cpu().numpy(), average='macro')
        #valid_acc = f1_score(labels[split_idx['valid']].detach().cpu().numpy(), y_pred[split_idx['valid']].detach().cpu().numpy(), average='macro')
        #test_acc = f1_score(labels[split_idx['test']].detach().cpu().numpy(), y_pred[split_idx['test']].detach().cpu().numpy(), average='macro')

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='GCN', help='GCN/SAGE')
    parser.add_argument('--metric', type=str, default='acc', help='acc/f1')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--cluster', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=40)
    parser.add_argument('--tau', type=float, default=1.0, help='softmax tempurate.')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dataset1 = PygNodePropPredDataset(name='ogbn-arxiv')
    data1 = dataset1[0]
    split_idx = dataset1.get_idx_split()
    data1.train_mask = split_idx['train']
    data1.valid_mask = split_idx['valid']
    data1.test_mask = split_idx['test']
    cluster_data = ClusterData(data1, num_parts=args.cluster, recursive=False, save_dir=dataset1.processed_dir)
    index = torch.zeros(data1.x.shape[0])
    for i in range(args.cluster):
        index[cluster_data.perm[cluster_data.partptr[i]:cluster_data.partptr[i+1]]] = i
    index = index.long().cuda()
    #cluster_train_label = one_hot(data.y.squeeze()[data1.train_mask], args.num_classes).cuda()
    #cluster_train_label = torch.mm(cluster_train_id.t(), cluster_train_label)
    #cluster_train_label = cluster_train_label / cluster_train_label.sum(1).unsqueeze(1)
    #import ipdb;ipdb.set_trace()
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    valid_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)
    #import ipdb;ipdb.set_trace()

    if args.encoder == 'SAGE':
        print("SAGE encoder")
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        print("GCN encoder")
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)
    
    test_res = torch.zeros(args.runs)
    test_res = test_res.cuda()
    for run in range(args.runs):
        model.reset_parameters()
        best_val_acc = 0
        best_test_acc = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, args, data, index, train_idx, optimizer)
            result = test(model, args, data, index, train_idx, valid_idx, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                if valid_acc > best_val_acc:
                    best_val_acc = valid_acc
                    best_test_acc = test_acc
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()