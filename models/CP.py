#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix


class CPGCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, ncluster, dropout=0.5, lr=0.01, weight_decay=5e-4 ,device=None):

        super(CPGCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass

        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.edge_index = None
        self.edge_weight = None
        self.features = None

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.fc1 = nn.Linear(nhid, nclass)
        self.fc2 = nn.Linear(nhid,ncluster)
        

    def forward(self, x, edge_index, edge_weight):

        x = F.relu(self.gc1(x, edge_index,edge_weight))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index,edge_weight)

        pred = self.fc1(x)
        pred_cluster = self.fc2(x)
        return pred, pred_cluster

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, adj, labels, clusters, idx_train, idx_val, lam=1, train_iters=200, verbose=False):
        """
        """
        self.initialize()

        self.edge_index, self.edge_weight = from_scipy_sparse_matrix(adj)
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.float().to(self.device)

        if sp.issparse(features):
            features = utils.sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        self.features = features.to(self.device)
        self.labels = torch.LongTensor(np.array(labels)).to(self.device)
        self.clusters = torch.LongTensor(np.array(clusters)).to(self.device)


        self._train_with_val(self.labels, self.clusters ,idx_train, idx_val, lam ,train_iters, verbose)


    def _train_with_val(self, labels, clusters, idx_train, idx_val, lam, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            pred, pred_cluster = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_pred = F.cross_entropy(pred[idx_train], labels[idx_train])
            loss_cluster = F.cross_entropy(pred_cluster[idx_train], clusters[idx_train])

            loss_train = loss_pred + lam * loss_cluster
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, pred loss: {:.4f}, cluster loss: {:.4f}'.format(i, loss_train, loss_cluster))

            self.eval()
            pred, pred_cluster = self.forward(self.features, self.edge_index, self.edge_weight)
            acc_val = utils.accuracy(pred[idx_val], labels[idx_val])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.pred = pred
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        pred, pred_cluster = self.forward(self.features, self.edge_index,self.edge_weight)
        loss_test = F.cross_entropy(pred[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(pred[idx_test], self.labels[idx_test])
        acc_cluster = utils.accuracy(pred_cluster[idx_test], self.clusters[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()),
              "cluster acc= {:.4f}".format(acc_cluster.item()))
        return pred

# %%
