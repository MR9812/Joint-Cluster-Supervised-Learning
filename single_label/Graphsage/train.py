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
from Data_air import Airports, AttributedGraphDataset
from sklearn.metrics import roc_auc_score, f1_score

    
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
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data):
    model.eval()
    pred, accs = model(data.x, data.adj_t), []
    loss = F.nll_loss(pred[data.val_mask], data.y[data.val_mask])
    pred = pred.argmax(dim=-1)
    idx_train = data.train_mask
    idx_val = data.val_mask
    idx_test = data.test_mask
    labels = data.y
    
    acc_train = int((pred[idx_train] == labels[idx_train]).sum()) / int(idx_train.sum())
    acc_val = int((pred[idx_val] == labels[idx_val]).sum()) / int(idx_val.sum())
    acc_test = int((pred[idx_test] == labels[idx_test]).sum()) / int(idx_test.sum())
    
    return acc_train, acc_val, acc_test, loss

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset', type=str, default='Cora', help='Cora/CiteSeer/PubMed/')
parser.add_argument('--runs', type=int, default=10, help='Runs')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--num_layers', type=int, default=2, help='Number of GCN layers.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
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
    best_val_loss = 999999999
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(model, optimizer, data)
        train_acc, val_acc, test_acc, val_loss = test(model, data)
        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc
            bad_counter = 0
        print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}, '
              f'Final Test: {best_test_acc:.4f}')
    print(run+1, " final acc：", best_test_acc)
    test_res[run] = best_test_acc
 
print("=== Final ===")
print(torch.max(test_res))
print(torch.min(test_res))
print("10 mean",torch.mean(test_res))
print("10 std",test_res.std())
print(test_res)
    