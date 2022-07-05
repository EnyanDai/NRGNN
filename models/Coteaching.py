#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import utils
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
from models.GCN import GCN

class Coteaching(nn.Module):
    """ 2 Layer Graph Convolutional Network.
    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,device=None):

        super(Coteaching, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

        self.output = None
        self.best_model = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None

        self.GCN1 = GCN(nfeat,nhid,nclass,dropout,device=device)
        self.GCN2 = GCN(nfeat,nhid,nclass,dropout,device=device)

    def forward(self, x, edge_index, edge_weight):

        return self.GCN1(x,edge_index,edge_weight), self.GCN2(x,edge_index,edge_weight)

    def fit(self, features, adj, labels, idx_train, idx_val=None, noise_rate=0.2, ek=10,train_iters=200, verbose=True):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        """

        self.edge_index, self.edge_weight = from_scipy_sparse_matrix(adj)
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.float().to(self.device)

        if sp.issparse(features):
            features = utils.sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        self.features = features.to(self.device)
        self.labels = torch.LongTensor(np.array(labels)).to(self.device)

        self.noise_rate = noise_rate
        self._train_with_val(self.labels, idx_train, idx_val, ek ,train_iters, verbose)

    def _train_with_val(self, labels, idx_train, idx_val, ek,train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        idx_train = np.asarray(idx_train)

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output1, output2 = self.forward(self.features, self.edge_index, self.edge_weight)

            pred_1 = output1[idx_train].max(1)[1]
            pred_2 = output2[idx_train].max(1)[1]


            disagree = (pred_1 != pred_2).cpu().numpy()
            idx_update = idx_train[disagree]

            if len(idx_update) == 0:
                break

            k = int((1 - min(i*self.noise_rate/ek, self.noise_rate)) * len(idx_update))

            loss_1 = F.cross_entropy(output1[idx_update], labels[idx_update], reduction='none')
            loss_2 = F.cross_entropy(output2[idx_update], labels[idx_update], reduction='none')

            _, topk_1 = torch.topk(loss_1, k, largest=False)
            _, topk_2 = torch.topk(loss_2, k, largest=False)

            loss_train = loss_1[topk_2].mean() + loss_2[topk_1].mean()

            loss_train.backward()
            optimizer.step()

            # if verbose and i % 10 == 0:

            self.eval()
            output1, output2 = self.forward(self.features, self.edge_index, self.edge_weight)
            acc_val = max(utils.accuracy(output1[idx_val], labels[idx_val]),utils.accuracy(output2[idx_val], labels[idx_val]))

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                weights = deepcopy(self.state_dict())
            if verbose and i % 1 == 0:
                print('Epoch {}, training loss: {}, acc_val: {:.4f}'.format(i, loss_train.item(),acc_val))

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
        output1, output2 = self.forward(self.features, self.edge_index, self.edge_weight)
        acc_1 = utils.accuracy(output1[idx_test], self.labels[idx_test])
        acc_2 = utils.accuracy(output2[idx_test], self.labels[idx_test])
        print("Test set results:",
              "acc_1= {:.4f}".format(acc_1.item()),
              "acc_2y= {:.4f}".format(acc_2.item()))
        return output1,output2


# %%
