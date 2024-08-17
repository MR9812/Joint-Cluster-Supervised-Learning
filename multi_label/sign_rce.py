#python sign_rce.py --dataset yelp --hidden_channels 512 --cluster 500
#python sign_rce.py --dataset amazon --hidden_channels 1024 --cluster 5000
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.transforms import SIGN

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from sklearn.metrics import f1_score
import torch_geometric as pyg
import random
import numpy as np


def one_hot(x, class_count):
    res = torch.nn.functional.one_hot(x, class_count)
    res = res.float()
    # res = torch.eye(class_count)
    # res = res[x,:]
    return res

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers + 1):
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.fc1 = torch.nn.Linear((num_layers + 1) * hidden_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels * 2, out_channels * 4)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.bn1.reset_parameters()

    def forward(self, xs, is_val = False, cluster_index = None):
        outs = []
        for x, lin, bn in zip(xs, self.lins, self.bns):
            x = bn(lin(x))
            x = F.relu(x)
            out = F.dropout(x, p=self.dropout, training=self.training)
            outs.append(out)
        x = torch.cat(outs, dim=-1)
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if is_val == False:
            cluster_features = torch.mm(cluster_index.t(), x) / cluster_index.sum(0).unsqueeze(1)
            x = torch.cat((torch.cat((x, cluster_features[cluster_index.argmax(-1)]), 1), torch.cat((cluster_features[cluster_index.argmax(-1)], x), 1)), 0)
        else:
            x = torch.cat((x, x.mean(0).unsqueeze(0).repeat(x.shape[0],1)), 1)
        x = self.fc2(x)
        return x


def train(model, train_loader, xs, y_true, optimizer, cluster_num_init, num_node, batch_size, device):
    model.train()
    num_classes = y_true.shape[1]
    
    for batch in train_loader:
        optimizer.zero_grad()
        y = y_true[batch]
        if batch.shape[0] == batch_size:
            cluster_num = cluster_num_init
        else:
            cluster_num = max(int(batch.shape[0]/num_node), 1) 
        index = random.sample(range(0, y.shape[0]), y.shape[0])
        index = torch.from_numpy(np.array(index)) % cluster_num
        cluster_index = one_hot(index, cluster_num).to(device)
        node_label = one_hot(y.long().reshape(1,-1).squeeze(), 2).to(device)
        cluster_label = torch.mm(cluster_index.t(), node_label.reshape(-1,num_classes*2))
        cluster_label = cluster_label / cluster_index.sum(0).unsqueeze(1)
        cluster_label = cluster_label[index].reshape(-1, 2)
        #rce_label = torch.matmul(node_label.unsqueeze(2), cluster_label.unsqueeze(1)).reshape(-1,4)
        rce_label1 = torch.matmul(node_label.unsqueeze(2), cluster_label.unsqueeze(1))
        rce_label2 = torch.matmul(cluster_label.unsqueeze(2), node_label.unsqueeze(1))
        rce_label = torch.cat((torch.matmul(node_label.unsqueeze(2), cluster_label.unsqueeze(1)), torch.matmul(cluster_label.unsqueeze(2), node_label.unsqueeze(1))), 0).reshape(-1, 4)
        y_pre = model([x[batch] for x in xs], False, cluster_index)
        y_pre = F.softmax(y_pre.reshape(-1,4), dim=1)
        y_pre = torch.log(y_pre + 1e-8)
        
        loss = -torch.sum(y_pre * rce_label) / y_pre.shape[0]
        loss.backward()
        optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, xs, y_true, cluster_num_init, num_node, batch_size):
    model.eval()

    y_preds = []
    num_classes = y_true.shape[1]
    loader = DataLoader(range(y_true.size(0)), batch_size=50000)
    for perm in loader:
        y_pred = model([x[perm] for x in xs], True)
        y_pred = F.softmax(y_pred.reshape(-1, num_classes, 4), dim=2)
        y_pred = y_pred.view(-1, num_classes, 2, 2).sum(dim=3)
        y_pred = y_pred.argmax(dim=-1, keepdim=True).squeeze()
        y_pred = y_pred.float()
        y_preds.append(y_pred.cpu())
    #import ipdb;ipdb.set_trace()
    y_pred = torch.cat(y_preds, dim=0)
    y_pred = y_pred.numpy()
    label = y_true.cpu().numpy()
    test_f1 = f1_score(label, y_pred, average='micro')

    return test_f1

def main():
    parser = argparse.ArgumentParser(description='SIGN')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=50000)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--cluster', type=int, default=1000)
    args = parser.parse_args()
    args.num_node = int(args.batch_size/args.cluster)
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
    y_train_true = data.y[train_mask].to(device)
    y_valid_true = data.y[val_mask].to(device)
    y_test_true = data.y[test_mask].to(device)
    data = SIGN(args.num_layers)(data)  # This might take a while.

    xs = [data.x] + [data[f'x{i}'] for i in range(1, args.num_layers + 1)]
    xs_train = [x[train_mask].to(device) for x in xs]
    xs_valid = [x[val_mask].to(device) for x in xs]
    xs_test = [x[test_mask].to(device) for x in xs]

    model = MLP(data.x.size(-1), args.hidden_channels, dataset.num_classes, args.num_layers, args.dropout).to(device)
    train_loader = torch.utils.data.DataLoader(range(train_mask.sum()), batch_size=args.batch_size, shuffle=True, drop_last=False)
    
    vals, tests = [], []
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val, final_test = 0, 0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, train_loader, xs_train, y_train_true, optimizer, args.cluster, args.num_node, args.batch_size, device)
            valid_acc = test(model, xs_valid, y_valid_true, args.cluster, args.num_node, args.batch_size)
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Valid: {100 * valid_acc:.2f}%, '
                  f'Final_test: {100 * final_test:.2f}%')
            if valid_acc > best_val:
               best_val = valid_acc
               final_test = test(model, xs_test, y_test_true, args.cluster, args.num_node, args.batch_size)

        print(f'Run{run} val:{best_val:.6f}, test:{final_test:.6f}')
        vals.append(best_val)
        tests.append(final_test)
    print('')
    print("test:", tests)
    print(f"Average val accuracy: {np.mean(vals)} ± {np.std(vals):.6f}")
    print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests):.6f}")
    print(args)


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