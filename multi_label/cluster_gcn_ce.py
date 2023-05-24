#python cluster_gcn_ce.py
#python cluster_gcn_ce.py --dataset amazon --num_partitions 6000 --eval_epo 19
import argparse

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborSampler
from torch_geometric.nn import SAGEConv
import numpy as np
import os

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric as pyg
import time
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import random

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, sigmoid):
        super(SAGE, self).__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.sigmoid = sigmoid
        self.output_dim = out_channels
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        if self.sigmoid:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if self.sigmoid:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        if self.sigmoid:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.sigmoid:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        if self.sigmoid:
            return x
        else:
            return torch.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if self.sigmoid and i != len(self.convs) - 1:
                    x = self.bns[i](x)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all





def train(model, loader, optimizer, device):
    model.train()

    total_loss = 0
    total_correct = total_examples = 0

    for data in loader:
        data = data.to(device)
        if data.train_mask.sum() == 0:
            continue
        optimizer.zero_grad()
        x = data.x

        out = model(x, data.edge_index)
        y = data.y
        if not model.sigmoid:
            loss = F.nll_loss(out[data.train_mask], y[data.train_mask])
        else:
            loss = torch.nn.BCEWithLogitsLoss()(out[data.train_mask], y[data.train_mask].to(torch.float))

        loss.backward()
        optimizer.step()

        num_examples = data.train_mask.sum().item()
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        if not model.sigmoid:
            total_correct += out[data.train_mask].argmax(dim=-1).eq(y[data.train_mask]).sum().item()
        else:
            total_correct += 0.

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def test(model, data, subgraph_loader, device):
    model.eval()

    out = model.inference(data.x, subgraph_loader, device)


    if not model.sigmoid:
        y_true = data.y
        y_pred = out.argmax(dim=-1)
        correct = y_pred.eq(y_true)
        train_acc = correct[data.train_mask].sum().item() / data.train_mask.sum().item()
        valid_acc = correct[data.val_mask].sum().item() / data.val_mask.sum().item()
        test_acc = correct[data.test_mask].sum().item() / data.test_mask.sum().item()
    else:
        out = (out > 0).float().numpy()
        label = data.y.numpy()
        train_acc = f1_score(label[data.train_mask], out[data.train_mask], average='micro')
        valid_acc = f1_score(label[data.val_mask], out[data.val_mask], average='micro')
        test_acc = f1_score(label[data.test_mask], out[data.test_mask], average='micro')

    return train_acc, valid_acc, test_acc



def main():
    parser = argparse.ArgumentParser(description='Cluster-GCN')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--dataset", type=str, default="yelp", help="The input dataset.")
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_epo', type=int, default=9)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_partitions', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()
    if args.dataset == 'yelp' or args.dataset == 'amazon':
        args.sigmoid = True
    else:
        args.sigmoid = False
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if args.dataset == 'yelp':
        print("load yelp")
        dataset = pyg.datasets.Yelp(root="./data/Yelp")
    else:
        print("load amazon")
        dataset = pyg.datasets.AmazonProducts(root="./data/amazon_products")
    
    data = dataset[0]
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    cluster_data = ClusterData(data, num_parts=args.num_partitions,
                               recursive=False, save_dir=dataset.processed_dir)
    loader = ClusterLoader(cluster_data, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)
    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                      batch_size=4096, shuffle=False,
                                      num_workers=args.num_workers)

    model = SAGE(dataset.num_features, args.hidden_channels, dataset.num_classes, args.num_layers, args.dropout, args.sigmoid).to(device)

    results_runs = []
    for run in range(args.runs):
        start_time = time.time()
        torch.cuda.synchronize()
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        final_test = 0
        best_valid_acc = 0
        for epoch in range(1, 1 + args.epochs):
            init_time = time.time()
            loss, train_acc = train(model, loader, optimizer, device)

            print(f'Run: {run + 1:02d}, 'f'Epoch: {epoch:02d}, 'f'Loss: {loss:.4f}, 'f'Approx Train Acc: {train_acc:.4f}')
            if epoch > args.eval_epo and epoch % args.eval_steps == 0 or epoch == args.epochs :
                train_acc, valid_acc, test_acc = test(model, data, subgraph_loader, device)
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train: {100 * train_acc:.3f}%, '
                      f'Valid: {100 * valid_acc:.3f}%, '
                      f'Test: {100 * test_acc:.3f}%')
                epoch_time = time.time() - init_time
                print("running time per epoch: ", epoch_time)
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    final_test = test_acc

        torch.cuda.synchronize()
        end_time = time.time()
        results_runs.append(final_test)
        print(f'Run{run} val:{best_valid_acc:.6f}, test:{final_test:.6f}')

    print(f"Average test accuracy: {np.mean(results_runs)} ± {np.std(results_runs):.6f}")



def set_seed():
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    set_seed()
    main()