import os.path as osp
from utils.args import base_args, use_smiles, use_smiles_edonicity, use_bbbp, use_AIDS
from torch_geometric.data import DataLoader as Dl
from tqdm import tqdm
import argparse
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GCNConv, GAE, VGAE

from torch_geometric.nn import SAGEConv, global_max_pool, GraphConv, TopKPooling, GCNConv, GatedGraphConv, GATConv,  TAGConv,  SGConv, global_mean_pool

from sklearn.metrics import roc_auc_score, average_precision_score


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GraphConv(in_channels, 2 * out_channels)
        self.conv2 = GraphConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def prepare_dataset(dataset, kwargs):
    data = dataset.dataset
    out=list()
    for el in tqdm(data) :
        #el.train_mask = el.val_mask = el.test_mask = el.y = None
        el = train_test_split_edges(el)
        #el.train_pos_edge_index = train_test_split_edges(el)
        out.append(el)
    init_fn = lambda x: np.random.seed(kwargs["seed"])
    return Dl(out, batch_size=kwargs["batch_size"], shuffle=True, worker_init_fn=init_fn)
    #return Dl(out, batch_size=1, shuffle=True, worker_init_fn=init_fn)


def gae_encode_experiment(**kwargs):

    num_features=kwargs["input_dim"]
    out_channels = 20
    model = GAE(GCNEncoder(num_features, out_channels))
    model = model.to(kwargs["device"])

    train_set = prepare_dataset(kwargs["trainset"], kwargs)
    test_set = prepare_dataset(kwargs["testset"], kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    auc, ap = 0, 0
    for epoch in range(1, kwargs["epochs"] + 1):
        loss = train(model, train_set, optimizer)
        print(loss)
        auc, ap = test(model,test_set)
        print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))


import math
def train(model, dataset, optimizer, variational=False):
    model.train()
    losses= list()
    for el in dataset:
        e = el.to(kwargs["device"])
        optimizer.zero_grad()
        z = model.encode(e.x, e.train_pos_edge_index)
        loss = model.recon_loss(z, e.train_pos_edge_index)
        if math.isnan(loss):
            continue
        if variational:
            loss = loss + (1 / e.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
    return float(sum(losses)/len(losses))


def test(model,dataset):#x,train_pos_edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    y = list()#torch.cat([], dim=0)
    pred = list()#torch.cat([], dim=0)
    with torch.no_grad():
        for el in dataset :
            e = el.to(kwargs["device"])
            z = model.encode(e.x, e.train_pos_edge_index)
            a, b = test_aux(z,model, e.test_pos_edge_index, e.test_neg_edge_index)
            y.append(a)
            pred.append(b)
    y = torch.cat(y, dim=0).numpy()
    pred = torch.cat(pred, dim =0).numpy()
    return roc_auc_score(y, pred), average_precision_score(y, pred)
    #return model.test(z, pos_edge_index, neg_edge_index)


def test_aux(z, model, pos_edge_index, neg_edge_index):
    #z = model.encode(el, el.train_pos_edge_index)

    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)

    pos_pred = model.decoder(z, pos_edge_index, sigmoid=True)
    neg_pred = model.decoder(z, neg_edge_index, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0)

    y, pred = y.detach().cpu(), pred.detach().cpu()
    return y, pred
kwargs = dict()
kwargs.update(base_args())
#kwargs["device"] = torch.device('cpu')

#kwargs.update(use_smiles(kwargs))
#kwargs.update(use_bbbp(kwargs))
kwargs.update(use_AIDS(kwargs))

gae_encode_experiment(**kwargs)