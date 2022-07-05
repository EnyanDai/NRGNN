#%%
import time
import argparse
import numpy as np
import torch
from models.GCN import GCN
from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.utils import preprocess

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=11, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
        choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed','dblp'], help='dataset')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train.')
parser.add_argument("--label_rate", type=float, default=0.05, help='rate of labeled data')
parser.add_argument('--ptb_rate', type=float, default=0.1, help="noise ptb_rate")
parser.add_argument('--noise', type=str, default='uniform', choices=['uniform', 'pair'], help='dataset')
parser.add_argument('--cluster', type=int, default=8)
parser.add_argument('--lam', type=float, default=0.2)
parser.add_argument('--density', type=float, default=0.2)
args = parser.parse_known_args()[0]

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

np.random.seed(15) 

if args.dataset=='dblp':
    from torch_geometric.datasets import CitationFull
    import torch_geometric.utils as utils
    dataset = CitationFull('./data','dblp')
    adj = utils.to_scipy_sparse_matrix(dataset.data.edge_index)
    features = dataset.data.x.numpy()
    labels = dataset.data.y.numpy()
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    idx_test = idx[:int(0.8 * len(labels))]
    idx_val = idx[int(0.8 * len(labels)):int(0.9 * len(labels))]
    idx_train = idx[int(0.9 * len(labels)):int((0.9+args.label_rate) * len(labels))]
else:
    data = Dataset(root='./data', name=args.dataset, setting='nettack')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_train = idx_train[:int(args.label_rate * adj.shape[0])]

from utils import load_emd
embedding = load_emd('./data', args.dataset)


# %%
from sklearn.cluster import KMeans

kmeans = KMeans(args.cluster, random_state=args.seed)
kmeans.fit(embedding)
community_labels = kmeans.labels_

# %%
#%%
from utils import noisify_with_P
ptb = args.ptb_rate
nclass = labels.max() + 1
train_labels = labels[idx_train]
noise_y, P = noisify_with_P(train_labels,nclass, ptb, 10, args.noise)
noise_labels = labels.copy()
noise_labels[idx_train] = noise_y

#noise_val_y,_ = noisify_with_P(labels[idx_val],nclass, ptb,args.seed)
#noise_labels[idx_val] = noise_val_y
from deeprobust.graph.global_attack import Random
import random
random.seed(15)
attacker = Random()
n_perturbations = int((1-args.density) * (adj.sum()//2))
attacker.attack(adj, n_perturbations, type='remove')
adj = attacker.modified_adj
#%%
np.random.seed(args.seed)
torch.manual_seed(args.seed)

from models.CP import CPGCN
model = CPGCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max() + 1,
            ncluster=community_labels.max() + 1,
            dropout=args.dropout, device=device).to(device)

model.fit(features, adj, noise_labels, community_labels, idx_train, idx_val, args.lam ,train_iters=args.epochs)
output = model.test(idx_test)