import argparse

import torch
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from arxiv_logger import Logger
import torch.nn as nn
import random
import numpy as np
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborSampler
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, dropout_adj
from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score, f1_score

def one_hot(x, class_count):
    res = torch.eye(class_count).cuda()
    return res[x,:]

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        #self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.fc1 = nn.Linear(hidden_channels * 2, out_channels * out_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.fc1.reset_parameters()

    def forward(self, x, cluster_id, cluster_index, is_val = False):
        for i, lin in enumerate(self.lins[:]):
            x = lin(x)
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


def train(model, x, y_true, index, train_idx, optimizer, args):
    model.train()
    optimizer.zero_grad()
    
    labels = y_true.squeeze(1)
    cluster_train_id = one_hot(index[train_idx], args.cluster).cuda()
    cluster_train_label = one_hot(labels[train_idx], args.num_classes).cuda()
    cluster_train_label = torch.mm(cluster_train_id.t(), cluster_train_label)
    cluster_train_label = cluster_train_label / cluster_train_label.sum(1).unsqueeze(1)
    
    y_pre = model(x, cluster_train_id, train_idx)
    y_pre = F.softmax(y_pre, dim=1)
    
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
def test(model, x, y_true, index, split_idx, evaluator, args, train_idx, valid_idx):
    model.eval()
    
    labels = y_true.squeeze(1)
    cluster_val_id = one_hot(index[torch.cat((train_idx, valid_idx),0)], args.cluster).cuda()
    cluster_val_label = one_hot(labels[torch.cat((train_idx, valid_idx),0)], args.num_classes).cuda()
    cluster_val_label = torch.mm(cluster_val_id.t(), cluster_val_label) 
    cluster_val_label = cluster_val_label / cluster_val_label.sum(1).unsqueeze(1)
    
    y_pre = model(x, one_hot(index, args.cluster).cuda(), torch.cat((train_idx, valid_idx),0), True)
    y_pre = F.softmax(y_pre, dim=1)
    y_pre = y_pre.view(-1, args.num_classes, args.num_classes)
    y_pre = y_pre.sum(dim=2)
    y_pred = y_pre.argmax(dim=-1, keepdim=True)

    if args.metric == 'acc':
        train_acc = evaluator.eval({
            'y_true': y_true[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': y_true[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': y_true[split_idx['test']],
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
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (MLP)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--cluster', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=40)
    parser.add_argument('--encoder', type=str, default='MLP', help='MLP/SGC')
    parser.add_argument('--metric', type=str, default='acc', help='acc/f1')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    #cluster
    dataset1 = PygNodePropPredDataset(name='ogbn-arxiv')
    data1 = dataset1[0]
    data1.edge_index = to_undirected(data1.edge_index, data1.num_nodes)
    split_idx = dataset1.get_idx_split()
    data1.train_mask = split_idx['train']
    data1.valid_mask = split_idx['valid']
    data1.test_mask = split_idx['test']
    cluster_data = ClusterData(data1, num_parts=args.cluster, recursive=False, save_dir=dataset1.processed_dir)
    index = torch.zeros(data1.x.shape[0])
    for i in range(args.cluster):
        index[cluster_data.perm[cluster_data.partptr[i]:cluster_data.partptr[i+1]]] = i
    index = index.long().cuda()
    
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    #split_idx = dataset.get_idx_split()
    data = dataset[0]
    x = data.x
    if args.encoder == 'SGC':
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)
        adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], sparse_sizes=(data.num_nodes, data.num_nodes))
        adj_t = adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        for i in range(args.num_layers):
            print("###")
            x = torch.mm(adj_t.to_dense(), x) 
        
    if args.use_node_embedding:
        embedding = torch.load('embedding.pt', map_location='cpu')
        x = torch.cat([x, embedding], dim=-1)
    x = x.to(device)

    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)
    valid_idx = split_idx['valid'].to(device)

    model = MLP(x.size(-1), args.hidden_channels, dataset.num_classes,
                args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, x, y_true, index, train_idx, optimizer, args)
            result = test(model, x, y_true, index, split_idx, evaluator, args, train_idx, valid_idx)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()