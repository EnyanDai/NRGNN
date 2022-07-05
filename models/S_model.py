#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from models.GCN import GCN
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
import utils

class NoiseAda(nn.Module):

    def __init__(self, class_size):
        super(NoiseAda, self).__init__()
        P = torch.FloatTensor(utils.build_uniform_P(class_size,0.1))
        self.B = torch.nn.parameter.Parameter(torch.log(P))
    
    def forward(self, pred):
        P = F.softmax(self.B, dim=1)
        return pred @ P


class S_model(GCN):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5,device=None):

        super(S_model, self).__init__(nfeat, nhid, nclass, device=device)
        self.noise_ada = NoiseAda(nclass)

    def fit(self, features, adj, labels, idx_train, idx_val=None,train_iters=200, verbose=False):

        self.device = self.gc1.weight.device
        self.initialize()

        self.edge_index, self.edge_weight = from_scipy_sparse_matrix(adj)
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.float().to(self.device)

        if sp.issparse(features):
            features = utils.sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        self.features = features.to(self.device)

        self.labels = torch.LongTensor(np.array(labels)).to(self.device)


        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()

            output = self.forward(self.features, self.edge_index, self.edge_weight)
            pred = F.softmax(output,dim=1)
            eps = 1e-8
            score = self.noise_ada(pred).clamp(eps,1-eps)
            
            loss_train = F.cross_entropy(torch.log(score[idx_train]), self.labels[idx_train])
            loss_train.backward()

            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            pred = F.softmax(output,dim=1)
            eps = 1e-8
            score = self.noise_ada(pred).clamp(eps,1-eps)
            
            loss_train = F.cross_entropy(torch.log(score[idx_train]), self.labels[idx_train])
            loss_train.backward()

            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_acc_val < acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
                if verbose:
                    print('=========save weights=========')
                    print("Epoch {}, val acc: {:.4f}".format(i,acc_val))

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
# %%
