#%%
import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import torch_geometric.utils as utils
import scipy.sparse as sp
from models.GCN import GCN
from utils import accuracy,sparse_mx_to_torch_sparse_tensor

class NRGNN:
    def __init__(self, args, device):

        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_acc_pred_val = 0
        self.best_pred = None
        self.best_graph = None
        self.best_model_index = None
        self.weights = None
        self.estimator = None
        self.model = None
        self.pred_edge_index = None

    def fit(self, features, adj, labels, idx_train, idx_val):

        args = self.args

        edge_index, _ = utils.from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)

        if sp.issparse(features):
            features = sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        features = features.to(self.device)
        labels = torch.LongTensor(np.array(labels)).to(self.device)

        self.edge_index = edge_index
        self.features = features
        self.labels = labels
        self.idx_unlabel = torch.LongTensor(list(set(range(features.shape[0])) - set(idx_train))).to(self.device)

        self.predictor = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         self_loop=True,
                         dropout=self.args.dropout, device=self.device).to(self.device)

        self.model = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         self_loop=True,
                         dropout=self.args.dropout, device=self.device).to(self.device)

        self.estimator = EstimateAdj(features.shape[1], args, idx_train ,device=self.device).to(self.device)
        
        # obtain the condidate edges linking unlabeled and labeled nodes
        self.pred_edge_index = self.get_train_edge(edge_index,features,self.args.n_p ,idx_train)

        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.estimator.parameters())+ list(self.predictor.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)

        # Train model
        t_total = time.time()
        for epoch in range(args.epochs):
            self.train(epoch, features, edge_index, idx_train, idx_val)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)
        self.predictor.load_state_dict(self.predictor_model_weigths)

        print("=====validation set accuracy=======")
        self.test(idx_val)
        print("===================================")

    def train(self, epoch, features, edge_index, idx_train, idx_val):
        args = self.args

        t = time.time()
        self.model.train()
        self.predictor.train()
        self.optimizer.zero_grad()

        # obtain representations and rec loss of the estimator
        representations, rec_loss = self.estimator(edge_index,features)
        
        # prediction of accurate pseudo label miner
        predictor_weights = self.estimator.get_estimated_weigths(self.pred_edge_index,representations)
        pred_edge_index = torch.cat([edge_index,self.pred_edge_index],dim=1)
        predictor_weights = torch.cat([torch.ones([edge_index.shape[1]],device=self.device),predictor_weights],dim=0)

        log_pred = self.predictor(features,pred_edge_index,predictor_weights)

        # obtain accurate pseudo labels and new candidate edges
        if self.best_pred == None:
            
            pred = F.softmax(log_pred,dim=1).detach()
            self.best_pred = pred
            self.unlabel_edge_index, self.idx_add = self.get_model_edge(self.best_pred)
        else:
            pred = self.best_pred

        # prediction of the GCN classifier
        estimated_weights = self.estimator.get_estimated_weigths(self.unlabel_edge_index,representations)
        estimated_weights = torch.cat([predictor_weights, estimated_weights],dim=0)
        model_edge_index = torch.cat([pred_edge_index,self.unlabel_edge_index],dim=1)
        output = self.model(features, model_edge_index, estimated_weights)
        pred_model = F.softmax(output,dim=1)

        eps = 1e-8
        pred_model = pred_model.clamp(eps, 1-eps)

        # loss from pseudo labels
        loss_add = (-torch.sum(pred[self.idx_add] * torch.log(pred_model[self.idx_add]), dim=1)).mean()
        
        # loss of accurate pseudo label miner
        loss_pred = F.cross_entropy(log_pred[idx_train], self.labels[idx_train])
        
        # loss of GCN classifier
        loss_gcn = F.cross_entropy(output[idx_train], self.labels[idx_train])

        total_loss = loss_gcn + loss_pred + self.args.alpha * rec_loss  + self.args.beta * loss_add
        total_loss.backward()

        self.optimizer.step()

        acc_train = accuracy(output[idx_train].detach(), self.labels[idx_train])

        # Evaluate validation set performance separately,
        self.model.eval()
        self.predictor.eval()
        pred = F.softmax(self.predictor(features,pred_edge_index,predictor_weights),dim=1)
        output = self.model(features, model_edge_index, estimated_weights.detach())

        acc_pred_val = accuracy(pred[idx_val], self.labels[idx_val])
        acc_val = accuracy(output[idx_val], self.labels[idx_val])

        if acc_pred_val > self.best_acc_pred_val:
            self.best_acc_pred_val = acc_pred_val
            self.best_pred_graph = predictor_weights.detach()
            self.best_pred = pred.detach()
            self.predictor_model_weigths = deepcopy(self.predictor.state_dict())
            self.unlabel_edge_index, self.idx_add = self.get_model_edge(pred)

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = estimated_weights.detach()
            self.best_model_index = model_edge_index
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: {:.4f}'.format(self.best_val_acc.item()))

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_pred: {:.4f}'.format(loss_pred.item()),
                      'loss_add: {:.4f}'.format(loss_add.item()),
                      'rec_loss: {:.4f}'.format(rec_loss.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()))
                print('Epoch: {:04d}'.format(epoch+1),
                        'acc_train: {:.4f}'.format(acc_train.item()),
                        'acc_val: {:.4f}'.format(acc_val.item()),
                        'acc_pred_val: {:.4f}'.format(acc_pred_val.item()),
                        'time: {:.4f}s'.format(time.time() - t))
                print('Size of add idx is {}'.format(len(self.idx_add)))


    def test(self, idx_test):
        """Evaluate the performance of ProGNN on test set
        """
        features = self.features
        labels = self.labels

        self.predictor.eval()
        estimated_weights = self.best_pred_graph
        pred_edge_index = torch.cat([self.edge_index,self.pred_edge_index],dim=1)
        output = self.predictor(features, pred_edge_index,estimated_weights)
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("\tPredictor results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        self.model.eval()
        estimated_weights = self.best_graph
        model_edge_index = self.best_model_index
        output = self.model(features, model_edge_index,estimated_weights)
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("\tGCN classifier results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        return float(acc_test)
    

    def get_train_edge(self, edge_index, features, n_p, idx_train):
        '''
        obtain the candidate edge between labeled nodes and unlabeled nodes based on cosine sim
        n_p is the top n_p labeled nodes similar with unlabeled nodes
        '''

        if n_p == 0:
            return None

        poten_edges = []
        if n_p > len(idx_train) or n_p < 0:
            for i in range(len(features)):
                indices = set(idx_train)
                indices = indices - set(edge_index[1,edge_index[0]==i])
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        else:
            for i in range(len(features)):
                sim = torch.div(torch.matmul(features[i],features[idx_train].T), features[i].norm()*features[idx_train].norm(dim=1))
                _,rank = sim.topk(n_p)
                if rank.max() < len(features) and rank.min() >= 0:
                    indices = idx_train[rank.cpu().numpy()]
                    indices = set(indices)
                else:
                    indices = set()
                indices = indices - set(edge_index[1,edge_index[0]==i])
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        poten_edges = torch.as_tensor(poten_edges).T
        poten_edges = utils.to_undirected(poten_edges,len(features)).to(self.device)

        return poten_edges

    def get_model_edge(self, pred):

        idx_add = self.idx_unlabel[(pred.max(dim=1)[0][self.idx_unlabel] > self.args.p_u)]

        row = self.idx_unlabel.repeat(len(idx_add))
        col = idx_add.repeat(len(self.idx_unlabel),1).T.flatten()
        mask = (row!=col)
        unlabel_edge_index = torch.stack([row[mask],col[mask]], dim=0)

        return unlabel_edge_index, idx_add
                        
#%%
class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, nfea, args, idx_train ,device='cuda'):
        super(EstimateAdj, self).__init__()
        
        self.estimator = GCN(nfea, args.edge_hidden, args.edge_hidden,dropout=0.0,device=device)
        self.device = device
        self.args = args
        self.representations = 0

    def forward(self, edge_index, features):

        representations = self.estimator(features,edge_index,\
                                        torch.ones([edge_index.shape[1]]).to(self.device).float())
        rec_loss = self.reconstruct_loss(edge_index, representations)

        return representations,rec_loss
    
    def get_estimated_weigths(self, edge_index, representations):

        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        output = torch.sum(torch.mul(x0,x1),dim=1)

        estimated_weights = F.relu(output)
        estimated_weights[estimated_weights < self.args.t_small] = 0.0

        return estimated_weights
    
    def reconstruct_loss(self, edge_index, representations):
        
        num_nodes = representations.shape[0]
        randn = utils.negative_sampling(edge_index,num_nodes=num_nodes, num_neg_samples=self.args.n_n*num_nodes)
        randn = randn[:,randn[0]<randn[1]]

        edge_index = edge_index[:, edge_index[0]<edge_index[1]]
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg = torch.sum(torch.mul(neg0,neg1),dim=1)

        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos = torch.sum(torch.mul(pos0,pos1),dim=1)

        rec_loss = (F.mse_loss(neg,torch.zeros_like(neg), reduction='sum') \
                    + F.mse_loss(pos, torch.ones_like(pos), reduction='sum')) \
                    * num_nodes/(randn.shape[1] + edge_index.shape[1]) 

        return rec_loss

# %%
