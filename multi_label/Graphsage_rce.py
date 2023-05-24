#python Graphsage_rce.py --dataset yelp
#python Graphsage_rce.py --dataset amazon --epoch 40 --eval_epo 20
import argparse
import torch
from tqdm import tqdm
import torch.nn.functional as F
#from torch_geometric.data import NeighborSampler
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import time
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import torch_geometric as pyg
import random

def one_hot(x, class_count):
    res = torch.eye(class_count)
    return res[x,:]

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, sigmoid):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.sigmoid = sigmoid
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        if self.sigmoid:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if self.sigmoid:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.fc1 = torch.nn.Linear(hidden_channels * 2, out_channels * 4)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.sigmoid:
            for bn in self.bns:
                bn.reset_parameters()
        self.fc1.reset_parameters()

    def forward(self, x, adjs, cluster_index):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            #if self.sigmoid and i != len(self.convs) - 1:
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        #import ipdb;ipdb.set_trace()    
        cluster_features = torch.mm(cluster_index.t(), x) / cluster_index.sum(0).unsqueeze(1)
        x1 = cluster_features[cluster_index.argmax(-1)]
        x = torch.cat((torch.cat((x, x1), 1), torch.cat((x1, x), 1)), 0)
        #x = torch.cat((x, x1), 1)
        x = self.fc1(x)
        if self.sigmoid:
            return x
        else:
            return torch.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                #if self.sigmoid and i != len(self.convs) - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                if i == self.num_layers - 1:
                    #x = torch.cat((torch.cat((x, x.mean(0).unsqueeze(0).repeat(x.shape[0],1)), 1),torch.cat((x, x.mean(0).unsqueeze(0).repeat(x.shape[0],1)), 1)),0)
                    x = torch.cat((x, x.mean(0).unsqueeze(0).repeat(x.shape[0],1)), 1)
                    x = self.fc1(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def train(model, x, y, args, num_classes, loader, optimizer, device, prin_feat=None):
    model.train()

    total_loss = total_correct = 0
    total_examples = 0
    for batch_size, n_id, adjs in loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        #import ipdb;ipdb.set_trace()
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()

        x_clone = x
        labels = y[n_id[:batch_size]]
        node_sum = labels.shape[0]
        if node_sum < args.batch_size:
            cluster_num = 1
        else:
            cluster_num = args.cluster
        index = random.sample(range(0, node_sum), node_sum)
        index = torch.from_numpy(np.array(index)) % cluster_num
        cluster_index = one_hot(index, cluster_num).cuda()
        node_label = one_hot(labels.long().reshape(1,-1).squeeze(), 2).cuda()
        cluster_label = torch.mm(cluster_index.t(), node_label.reshape(-1,num_classes*2))
        cluster_label = cluster_label / cluster_index.sum(0).unsqueeze(1)
        cluster_label = cluster_label[index].reshape(-1, 2)
        #rce_label = torch.matmul(node_label.unsqueeze(2), cluster_label.unsqueeze(1)).reshape(-1,4)
        rce_label1 = torch.matmul(node_label.unsqueeze(2), cluster_label.unsqueeze(1))
        rce_label2 = torch.matmul(cluster_label.unsqueeze(2), node_label.unsqueeze(1))
        rce_label = torch.cat((rce_label1, rce_label2), 0).reshape(-1, 4)
        y_pre = model(x_clone[n_id], adjs, cluster_index)
        y_pre = F.softmax(y_pre.reshape(-1,4), dim=1)
        y_pre = torch.log(y_pre + 1e-8)
    
        loss = -torch.sum(y_pre * rce_label) / y_pre.shape[0]

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item() * batch_size)
        total_examples += batch_size
        if not model.sigmoid:
            total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        else:
            total_correct += 0.

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def test(model, x, y, num_classes, train_mask, val_mask, test_mask, subgraph_loader, device):
    model.eval()

    y_pre = model.inference(x, subgraph_loader, device)
    if not model.sigmoid:
        pred = out.argmax(dim=-1)
        y_true = y.cpu()
        correct = pred.eq(y_true)
        train_acc = correct[train_mask].sum().item() / train_mask.sum().item()
        valid_acc = correct[val_mask].sum().item() / val_mask.sum().item()
        test_acc = correct[test_mask].sum().item() / test_mask.sum().item()
    else:
        y_pre = F.softmax(y_pre.reshape(-1, num_classes, 4), dim=2)
        y_pre = y_pre.view(-1, num_classes, 2, 2).sum(dim=3)
        #y_pre1 = y_pre[:int(y_pre.shape[0]/2)].view(-1, num_classes, 2, 2).sum(dim=3)
        #y_pre2 = y_pre[int(y_pre.shape[0]/2):].view(-1, num_classes, 2, 2).sum(dim=2)
        #y_pre = (y_pre1 + y_pre2) / 2
        y_pre = y_pre.argmax(dim=-1, keepdim=True).squeeze()
        out = y_pre.float().numpy()
        label = y.cpu().numpy()
        train_acc = f1_score(label[train_mask], out[train_mask], average='micro')
        valid_acc = f1_score(label[val_mask], out[val_mask], average='micro')
        test_acc = f1_score(label[test_mask], out[test_mask], average='micro')

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='GraphSAGE)')
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cluster', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--eval_epo', type=int, default=-1)
    parser.add_argument('--eval_steps', type=int, default=2)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--sigmoid', action='store_true',
                        help='True: multi-class classification')
    args = parser.parse_args()
    args.sigmoid = True
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    if args.dataset == 'yelp':
        dataset = pyg.datasets.Yelp(root="./data/Yelp")
    else:
        dataset = pyg.datasets.AmazonProducts(root="./data/amazon_products")
    data = dataset[0]
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=[15, 10], batch_size=args.batch_size,
                                   shuffle=True, num_workers=12)

    test_batch_size = 512
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                      batch_size=test_batch_size, shuffle=False,
                                      num_workers=12)
    model = SAGE(dataset.num_features, args.hidden_channels, dataset.num_classes,
                args.num_layers, args.dropout, args.sigmoid).to(device)
    x = data.x.to(device)
    y = data.y.squeeze().to(device)
    vals, tests = [], []
    for run in range(args.runs):
        start_time = time.time()
        torch.cuda.synchronize()
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val, final_test = 0, 0
        
        for epoch in range(1, 1 + args.epochs):
            init_time = time.time()
            loss, train_acc = train(model, x, y, args, dataset.num_classes, train_loader, optimizer, device)
            print(f'Run: {run + 1:02d}, '
                   f'Epoch: {epoch:02d}, '
                   f'Loss: {loss:.4f}, '
                   f'Approx Train Acc: {train_acc:.4f}')
            if epoch>args.eval_epo:
                train_acc, valid_acc, test_acc = test(model, x, y, dataset.num_classes, data.train_mask, data.val_mask, data.test_mask, subgraph_loader, device)
                epoch_time = time.time() - init_time
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train: {100 * train_acc:.3f}%, '
                      f'Valid: {100 * valid_acc:.3f}% '
                      f'Test: {100 * test_acc:.3f}%',
                      f'time: {epoch_time}')
                if valid_acc > best_val:
                    best_val = valid_acc
                    final_test = test_acc

        print(f'Run{run} val:{best_val:.6f}, test:{final_test:.6f}')
        vals.append(best_val)
        tests.append(final_test)

        torch.cuda.synchronize()
        end_time = time.time()
        
    print('')
    print("test:", tests)
    print(f"Average val accuracy: {np.mean(vals)} ± {np.std(vals):.6f}")
    print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests):.6f}")
    print(args)
        


def set_seed():
    seed = 123
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    set_seed()
    main()



