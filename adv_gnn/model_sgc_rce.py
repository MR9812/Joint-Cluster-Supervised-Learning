"""
Extended from https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/citation
"""
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from torch_geometric.nn import SGConv

class SGC(torch.nn.Module):
    """ SGC based on pytorch geometric. Simplifying Graph Convolutional Networks.
    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nclass : int
        size of output dimension
    K: int
        number of propagation in SGC
    cached : bool
        whether to set the cache flag in SGConv
    lr : float
        learning rate for SGC
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN.
        When `with_relu` is True, `weight_decay` will be set to 0.
    with_bias: bool
        whether to include bias term in SGC weights.
    device: str
        'cpu' or 'cuda'.
    Examples
    --------
	We can first load dataset and then train SGC.
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import SGC
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> sgc = SGC(nfeat=features.shape[1], K=3, lr=0.1,
              nclass=labels.max().item() + 1, device='cuda')
    >>> sgc = sgc.to('cuda')
    >>> pyg_data = Dpr2Pyg(data) # convert deeprobust dataset to pyg dataset
    >>> sgc.fit(pyg_data, train_iters=200, patience=200, verbose=True) # train with earlystopping
    """


    def __init__(self, nfeat, nclass, cluster, nhid = 64, K=2, cached=True, lr=0.01,
            weight_decay=5e-4, with_bias=True, device=None):

        super(SGC, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.conv1 = SGConv(nfeat, nhid, bias=with_bias, K=K, cached=cached)
        self.fc = nn.Linear(nhid * 2, nclass * nclass)

        self.weight_decay = weight_decay
        self.lr = lr
        self.cluster = cluster
        self.nclass = nclass
        self.output = None
        self.best_model = None
        self.best_output = None

    def forward(self, data, cluster_id, cluster_index, is_val = False):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, 0.5, training=self.training)
        
        if is_val == False:
            cluster_features = torch.mm(cluster_id.t(), x[cluster_index]) / cluster_id.sum(0).unsqueeze(1)
            x1 = cluster_features[cluster_id.argmax(1)]
            x = torch.cat((torch.cat((x[cluster_index], x1), 1), torch.cat((x1, x[cluster_index]), 1)), 0)
        else:
            cluster_features = torch.mm(cluster_id[cluster_index].t(), x[cluster_index]) / cluster_id[cluster_index].sum(0).unsqueeze(1)
            x1 = cluster_features[cluster_id.argmax(1)]
            x = torch.cat((x, x1), 1)
        
        x = self.fc(x)
        return x
    
    def one_hot(self, x, class_count):
        res = torch.eye(class_count)
        return res[x,:]

    def initialize(self):
        """Initialize parameters of SGC.
        """
        self.conv1.reset_parameters()
        self.fc.reset_parameters()

    def fit(self, pyg_data, index, train_iters=200, initialize=True, verbose=False, patience=500, **kwargs):
        """Train the SGC model, when idx_val is not None, pick the best model
        according to the validation loss.
        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        # self.device = self.conv1.weight.device
        if initialize:
            self.initialize()

        self.data = pyg_data[0].to(self.device)
        # By default, it is trained with early stopping on validation
        self.train_with_early_stopping(index, train_iters, patience, verbose)

    def train_with_early_stopping(self, index, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            print('=== training SGC model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask
        #import ipdb;ipdb.set_trace()
        cluster_train_id = self.one_hot(index[train_mask], self.cluster).cuda()
        cluster_train_label = self.one_hot(labels[train_mask], self.nclass).cuda()
        cluster_train_label = torch.mm(cluster_train_id.t(), cluster_train_label)
        cluster_train_label = cluster_train_label / cluster_train_label.sum(1).unsqueeze(1)
        
        cluster_val_id = self.one_hot(index[train_mask+val_mask], self.cluster).cuda()
        cluster_val_label = self.one_hot(labels[train_mask+val_mask], self.nclass).cuda()
        cluster_val_label = torch.mm(cluster_val_id.t(), cluster_val_label) 
        cluster_val_label = cluster_val_label / cluster_val_label.sum(1).unsqueeze(1)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            #relation ce : ground truth
            node_label = self.one_hot(labels[train_mask],  self.nclass).cuda()
            cluster_label = cluster_train_label[index[train_mask]].cuda()
            rce_label = torch.cat((torch.matmul(node_label.unsqueeze(2), cluster_label.unsqueeze(1)), torch.matmul(cluster_label.unsqueeze(2), node_label.unsqueeze(1))), 0)
            rce_label = rce_label.view(-1, self.nclass * self.nclass)
            
            y_pre = self.forward(self.data, cluster_train_id, train_mask)
            y_pre = F.softmax(y_pre, dim=1)

            rce_pre = torch.log(y_pre + 1e-8)
            loss_train = -torch.sum(rce_pre * rce_label) / rce_pre.shape[0]
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            node_label = self.one_hot(labels[val_mask],  self.nclass).cuda()
            cluster_label = cluster_val_label[index[val_mask]].cuda()
            rce_label = torch.matmul(node_label.unsqueeze(2), cluster_label.unsqueeze(1))
            rce_label = rce_label.view(-1, self.nclass * self.nclass)
            y_pre = self.forward(self.data, self.one_hot(index, self.cluster).cuda(), train_mask+val_mask, True)
            y_pre = F.softmax(y_pre, dim=1)

            rce_pre = torch.log(y_pre[val_mask] + 1e-8)
            loss_val = -torch.sum(rce_pre * rce_label) / rce_pre.shape[0]

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                #self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, index):
        """Evaluate SGC performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        train_mask = self.data.train_mask
        val_mask = self.data.val_mask
        test_mask = self.data.test_mask
        labels = self.data.y
        cluster_val_id = self.one_hot(index[train_mask+val_mask], self.cluster).cuda()
        cluster_val_label = self.one_hot(labels[train_mask+val_mask], self.nclass).cuda()
        cluster_val_label = torch.mm(cluster_val_id.t(), cluster_val_label) 
        cluster_val_label = cluster_val_label / cluster_val_label.sum(1).unsqueeze(1)
        
        y_pre = self.forward(self.data, self.one_hot(index, self.cluster).cuda(), train_mask+val_mask, True)
        y_pre = F.softmax(y_pre, dim=1)
        y_pre = y_pre.view(-1, self.nclass, self.nclass)
        y_pre = y_pre.sum(dim=2)
        
        #output = self.forward(self.data)
        # output = self.output
        #loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(y_pre[test_mask], labels[test_mask])
        print("Test set results:",
              #"loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def predict(self):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of SGC
        """

        self.eval()
        return self.forward(self.data)


if __name__ == "__main__":
    from deeprobust.graph.data import Dataset, Dpr2Pyg
    # from deeprobust.graph.defense import SGC
    data = Dataset(root='/tmp/', name='cora')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    sgc = SGC(nfeat=features.shape[1],
          nclass=labels.max().item() + 1, device='cpu')
    sgc = sgc.to('cpu')
    pyg_data = Dpr2Pyg(data)
    sgc.fit(pyg_data, verbose=True) # train with earlystopping
    sgc.test()
    print(sgc.predict())