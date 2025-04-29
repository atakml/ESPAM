# Import packages
import random
import time
from copy import deepcopy, copy
from itertools import combinations
from math import sqrt
import statistics

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.special
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, MessagePassing
from torch_geometric.utils import (dense_to_sparse, k_hop_subgraph, subgraph,
                                   to_dense_adj, to_networkx)
from tqdm import tqdm
from abc import ABC, abstractmethod
import networkx

class GCN(nn.Module):
    """
    Construct a GNN with several Graph Convolution blocks
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()

        self.dropout = dropout
        self.conv_in = GCNConv(input_dim, hidden_dim[0])
        self.conv = [GCNConv(hidden_dim[i - 1], hidden_dim[i])
                     for i in range(1, len(hidden_dim))]
        self.conv_out = GCNConv(hidden_dim[-1], output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv_in(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for block in self.conv:
            x = F.relu(block(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv_out(x, edge_index)

        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    """
    Contruct a GNN with several Graph Attention layers
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout, n_heads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.conv_in = GATConv(
            input_dim, hidden_dim[0], heads=n_heads[0], dropout=self.dropout)
        self.conv = [GATConv(hidden_dim[i - 1] * n_heads[i - 1], hidden_dim[i],
                             heads=n_heads[i], dropout=self.dropout) for i in range(1, len(n_heads) - 1)]
        self.conv_out = GATConv(
            hidden_dim[-1] * n_heads[-2], output_dim, heads=n_heads[-1], dropout=self.dropout, concat=False)

    def forward(self, x, edge_index, att=None):
        x = F.dropout(x, p=self.dropout, training=self.training)

        if att:  # if we want to see attention weights
            x, alpha = self.conv_in(
                x, edge_index, return_attention_weights=att)
            x = F.elu(x)

            for attention in self.conv:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.elu(attention(x, edge_index))

            x = F.dropout(x, p=self.dropout, training=self.training)
            x, alpha2 = self.conv_out(
                x, edge_index, return_attention_weights=att)

            return F.log_softmax(x, dim=1), alpha, alpha2

        else:  # we don't consider attention weights
            x = self.conv_in(x, edge_index)
            x = F.elu(x)

            for attention in self.conv:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.elu(attention(x, edge_index))

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv_out(x, edge_index)

            return F.log_softmax(x, dim=1)


class LinearRegressionModel(nn.Module):
    """Construct a simple linear regression
    """

    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred = self.linear1(x)
        return y_pred


class GCNNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, add_self=False, args=None):
        super(GCNNet, self).__init__()
        self.input_dim = input_dim
        print('GCNNet input_dim:', self.input_dim)
        self.hidden_dim = hidden_dim
        print('GCNNet hidden_dim:', self.hidden_dim)
        self.label_dim = label_dim
        print('GCNNet label_dim:', self.label_dim)
        self.num_layers = num_layers
        print('GCNNet num_layers:', self.num_layers)

        self.args = args
        self.dropout = dropout
        self.act = F.relu
        # self.celloss = torch.nn.CrossEntropyLoss()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(self.input_dim, self.hidden_dim))
        for layer in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
        print('len(self.convs):', len(self.convs))

        self.linear = torch.nn.Linear(
            len(self.convs) * self.hidden_dim, self.label_dim)

        # Init weights
        # torch.nn.init.xavier_uniform_(self.linear.weight.data)
        # for conv in self.convs:
        #    torch.nn.init.xavier_uniform_(conv.weight.data)

    def forward(self, x, edge_index):
        x_all = []

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_all.append(x)
        x = torch.cat(x_all, dim=1)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            add_self=False,
            normalize_embedding=False,
            dropout=0.0,
            bias=True,
            gpu=True,
            att=False,
    ):
        super(GraphConv, self).__init__()
        self.att = att
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        if not gpu:
            self.weight = nn.Parameter(
                torch.FloatTensor(input_dim, output_dim))
            if add_self:
                self.self_weight = nn.Parameter(
                    torch.FloatTensor(input_dim, output_dim)
                )
            if att:
                self.att_weight = nn.Parameter(
                    torch.FloatTensor(input_dim, input_dim))
        else:
            self.weight = nn.Parameter(
                torch.FloatTensor(input_dim, output_dim).cuda())
            if add_self:
                self.self_weight = nn.Parameter(
                    torch.FloatTensor(input_dim, output_dim).cuda()
                )
            if att:
                self.att_weight = nn.Parameter(
                    torch.FloatTensor(input_dim, input_dim).cuda()
                )
        if bias:
            if not gpu:
                self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            else:
                self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)

        if self.att:
            x_att = torch.matmul(x, self.att_weight)
            att = x_att @ x_att.permute(0, 2, 1)
            adj = adj * att

        y = torch.matmul(adj, x)
        y = torch.matmul(y, self.weight)
        if self.add_self:
            self_emb = torch.matmul(x, self.self_weight)
            y += self_emb
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)

        return y, adj


class GcnEncoderGraph(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            embedding_dim,
            label_dim,
            num_layers,
            pred_hidden_dims=[],
            concat=True,
            bn=True,
            dropout=0.0,
            add_self=False,
            args=None,
    ):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = add_self
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1
        self.celloss = nn.CrossEntropyLoss()

        self.bias = True
        self.gpu = args.gpu
        if args.method == "att":
            self.att = True
        else:
            self.att = False
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim,
            hidden_dim,
            embedding_dim,
            num_layers,
            add_self,
            normalize=True,
            dropout=dropout,
        )
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(
            self.pred_input_dim, pred_hidden_dims, label_dim, num_aggs=self.num_aggs
        )

        for m in self.modules():
            if isinstance(m, GraphConv):
                init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu"))
                if m.att:
                    init.xavier_uniform_(
                        m.att_weight.data, gain=nn.init.calculate_gain("relu")
                    )
                if m.add_self:
                    init.xavier_uniform_(
                        m.self_weight.data, gain=nn.init.calculate_gain("relu")
                    )
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    def build_conv_layers(
            self,
            input_dim,
            hidden_dim,
            embedding_dim,
            num_layers,
            add_self,
            normalize=False,
            dropout=0.0,
    ):
        conv_first = GraphConv(
            input_dim=input_dim,
            output_dim=hidden_dim,
            add_self=add_self,
            normalize_embedding=normalize,
            bias=self.bias,
            gpu=self.gpu,
            att=self.att,
        )
        conv_block = nn.ModuleList(
            [
                GraphConv(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    add_self=add_self,
                    normalize_embedding=normalize,
                    dropout=dropout,
                    bias=self.bias,
                    gpu=self.gpu,
                    att=self.att,
                )
                for i in range(num_layers - 2)
            ]
        )
        conv_last = GraphConv(
            input_dim=hidden_dim,
            output_dim=embedding_dim,
            add_self=add_self,
            normalize_embedding=normalize,
            bias=self.bias,
            gpu=self.gpu,
            att=self.att,
        )
        return conv_first, conv_block, conv_last

    def build_pred_layers(
            self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1
    ):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        """ For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        """
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, : batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        """ Batch normalization of 3D tensor x
        """
        bn_module = nn.BatchNorm1d(x.size()[1])
        if self.gpu:
            bn_module = bn_module.cuda()
        return bn_module(x)

    def gcn_forward(
            self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None
    ):
        """ Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
            The embedding dim is self.pred_input_dim
        """

        x, adj_att = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        adj_att_all = [adj_att]
        # out_all = []
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        for i in range(len(conv_block)):
            x, _ = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
            adj_att_all.append(adj_att)
        x, adj_att = conv_last(x, adj)
        x_all.append(x)
        adj_att_all.append(adj_att)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        self.embedding_tensor = x_tensor

        # adj_att_tensor: [batch_size x num_nodes x num_nodes x num_gc_layers]
        adj_att_tensor = torch.stack(adj_att_all, dim=3)
        return x_tensor, adj_att_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(
                max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x, adj_att = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        adj_att_all = [adj_att]
        for i in range(self.num_layers - 2):
            x, adj_att = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
            adj_att_all.append(adj_att)
        x, adj_att = self.conv_last(x, adj)
        adj_att_all.append(adj_att)
        # x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out

        # adj_att_tensor: [batch_size x num_nodes x num_nodes x num_gc_layers]
        adj_att_tensor = torch.stack(adj_att_all, dim=3)

        self.embedding_tensor = output
        ypred = self.pred_model(output)

        return F.log_softmax(ypred, dim=1)


class GcnEncoderNode(GcnEncoderGraph):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            embedding_dim,
            label_dim,
            num_layers,
            pred_hidden_dims=[],
            concat=True,
            bn=True,
            dropout=0.0,
            args=None,
    ):
        super(GcnEncoderNode, self).__init__(
            input_dim,
            hidden_dim,
            embedding_dim,
            label_dim,
            num_layers,
            pred_hidden_dims,
            concat,
            bn,
            dropout,
            args=args,
        )

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        # Convert ajd matrix
        adj = to_dense_adj(adj, max_num_nodes=x.shape[0])

        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(
                max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        self.adj_atts = []
        self.embedding_tensor, adj_att = self.gcn_forward(
            x, adj, self.conv_first, self.conv_block, self.conv_last, embedding_mask
        )
        pred = self.pred_model(self.embedding_tensor)
        pred = pred.squeeze(0)

        return F.log_softmax(pred, dim=1)  # pred

    def loss(self, output, target):
        return self.celloss(output, target)



























def plot_dist(noise_feats, label=None, ymax=1.1, color=None, title=None, save_path=None):
    """
    Kernel density plot of the number of noisy features included in explanations,
    for a certain number of test samples
    """
    if not any(noise_feats):  # handle special case where noise_feats=0
        noise_feats[0] = 0.25

    # plt.switch_backend("agg")
    sns.set_style('darkgrid')
    # sns.set_context("talk")
    ax = sns.distplot(noise_feats, hist=False, kde=True,
                      kde_kws={'label': label}, color=color)
    sns.set(font_scale=1.5)  # , rc={"lines.linewidth": 2})
    plt.xlim(-3, 8)
    plt.ylim(ymin=0.0, ymax=ymax)

    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path)

    return ax


def __flow__(model):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            return module.flow
    return 'source_to_target'


def visualize_subgraph(model, node_idx, edge_index, edge_mask, num_hops, y=None,
                       threshold=None, **kwargs):
    """Visualizes the subgraph around :attr:`node_idx` given an edge mask
    :attr:`edge_mask`.
    Args:
            node_idx (int): The node id to explain.
            edge_index (LongTensor): The edge indices - adj matrix
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                    as node colorings. (default: :obj:`None`)
            threshold (float, optional): Sets a threshold for visualizing
                    important edges. If set to :obj:`None`, will visualize all
                    edges with transparancy indicating the importance of edges.
                    (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                    :func:`nx.draw`.
    :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
    """
    assert edge_mask.size(0) == edge_index.size(1)
    edge_index = edge_index[:, edge_mask.nonzero().T[0]]

    # Only operate on a k-hop subgraph around `node_idx`.
    subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
        node_idx, num_hops, edge_index, relabel_nodes=True,
        num_nodes=None, flow=__flow__(model))

    # edge_mask = edge_mask[hard_edge_mask]

    if threshold is not None:
        edge_mask = (edge_mask >= threshold).to(torch.float)

    if y is None:
        y = torch.zeros(edge_index.max().item() + 1,
                        device=edge_index.device)
    else:
        y = y[subset].to(torch.float) / y.max().item()

    data = Data(edge_index=edge_index, att=edge_mask[edge_mask != 0], y=y,
                num_nodes=y.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
    mapping = {k: i for k, i in enumerate(subset.tolist())}
    G = nx.relabel_nodes(G, mapping)

    node_kwargs = copy(kwargs)
    node_kwargs['node_size'] = kwargs.get('node_size') or 800
    node_kwargs['cmap'] = kwargs.get('cmap') or 'cool'

    label_kwargs = copy(kwargs)
    label_kwargs['font_size'] = kwargs.get('font_size') or 10

    pos = nx.spring_layout(G)
    plt.switch_backend("agg")
    ax = plt.gca()
    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->",
                alpha=max(data['att'] * 2, 0.05),
                shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ))
    nx.draw_networkx_nodes(G, pos, node_color=y.tolist(), **node_kwargs)
    nx.draw_networkx_labels(G, pos, **label_kwargs)

    return ax, G


def denoise_graph(data, weighted_edge_mask, node_explanations, neighbours, node_idx, feat=None, label=None,
                  threshold_num=10):
    """Cleaning a graph by thresholding its node values.
    Args:
            - weighted_edge_mask:  Edge mask, with importance given to each edge
            - node_explanations :  Shapley values for neighbours
            - neighbours
            - node_idx          :  Index of node to highlight (TODO ?)
            - feat              :  An array of node features.
            - label             :  A list of node labels.
            - theshold_num      :  The maximum number of nodes to threshold.
    """
    # Subgraph with only relevant nodes - pytorch
    edge_index = data.edge_index[:, weighted_edge_mask.nonzero().T[0]]

    s = subgraph(
        torch.cat((torch.tensor([node_idx]), neighbours)), edge_index)[0]

    # Disregard size of explanations
    node_explanations = np.abs(node_explanations)

    # Create graph of neighbourhood of node of interest
    G = nx.DiGraph()
    G.add_nodes_from(neighbours.detach().numpy())
    G.add_node(node_idx)
    G.nodes[node_idx]["self"] = 1
    if feat is not None:
        for node in G.nodes():
            G.nodes[node]["feat"] = feat[node].detach().numpy()
    if label is not None:
        for node in G.nodes():
            G.nodes[node]["label"] = label[node].item()

    # Find importance threshold required to retrieve 10 most import nei.
    threshold_num = min(len(neighbours), threshold_num)
    threshold = np.sort(
        node_explanations)[-threshold_num]

    # Add edges
    # weighted_edge_list = [
    #      (data.edge_index[0, i].item(),
    #       data.edge_index[1, i].item(), weighted_edge_mask[i].item())
    #      for i, _ in enumerate(weighted_edge_mask)
    # if weighted_edge_mask[i] >= threshold
    #  ]

    # # Keep edges that satisfy the threshold
    # node_expl_dico = {}
    # for i, imp in enumerate(node_explanations):
    #     node_expl_dico[neighbours[i].item()] = imp
    # node_expl_dico[node_idx]=torch.tensor(0)
    # weighted_edge_list = [ (el1.item(),el2.item(),node_expl_dico[el1.item()].item()) for el1,el2 in zip(s[0],s[1]) ]

    # Add edges
    imp = weighted_edge_mask[weighted_edge_mask != 0]
    weighted_edge_list = [(el1.item(), el2.item(), i.item()) for el1, el2, i in (zip(s[0], s[1], imp))]
    G.add_weighted_edges_from(weighted_edge_list)

    # Keep nodes that satisfy the threshold
    del_nodes = []
    for i, node in enumerate(G.nodes()):
        if node != node_idx:
            if node_explanations[i] < threshold:
                del_nodes.append(node)
    G.remove_nodes_from(del_nodes)

    return G


def log_graph(G,
              identify_self=True,
              nodecolor="label",
              epoch=0,
              fig_size=(4, 3),
              dpi=300,
              label_node_feat=False,
              edge_vmax=None,
              args=None):
    """
    Args:
            nodecolor: the color of node, can be determined by 'label', or 'feat'. For feat, it needs to
                    be one-hot'
    """
    cmap = plt.get_cmap("Set1")
    plt.switch_backend("agg")
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    node_colors = []
    # edge_colors = [min(max(w, 0.0), 1.0) for (u,v,w) in G.edges.data('weight', default=1)]
    edge_colors = [w for (u, v, w) in G.edges.data("weight", default=1)]

    # maximum value for node color
    vmax = 8
    for i in G.nodes():
        if nodecolor == "feat" and "feat" in G.nodes[i]:
            num_classes = G.nodes[i]["feat"].size()[0]
            if num_classes >= 10:
                cmap = plt.get_cmap("tab20")
                vmax = 19
            elif num_classes >= 8:
                cmap = plt.get_cmap("tab10")
                vmax = 9
            break

    feat_labels = {}
    for i in G.nodes():
        if identify_self and "self" in G.nodes[i]:
            node_colors.append(0)
        elif nodecolor == "label" and "label" in G.nodes[i]:
            node_colors.append(G.nodes[i]["label"] + 1)
        elif nodecolor == "feat" and "feat" in G.nodes[i]:
            # print(G.nodes[i]['feat'])
            feat = G.nodes[i]["feat"].detach().numpy()
            # idx with pos val in 1D array
            feat_class = 0
            for j in range(len(feat)):
                if feat[j] == 1:
                    feat_class = j
                    break
            node_colors.append(feat_class)
            feat_labels[i] = feat_class
        else:
            node_colors.append(1)
    if not label_node_feat:
        feat_labels = None

    plt.switch_backend("agg")
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    # pos_layout = nx.kamada_kawai_layout(G, weight=None)
    pos_layout = nx.fruchterman_reingold_layout(G)

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        edge_vmax = 1
        edge_vmin = 0
    else:
        weights = [d for (u, v, d) in G.edges(data="weight", default=1)]
        if edge_vmax is None:
            edge_vmax = statistics.median_high(
                [d for (u, v, d) in G.edges(data="weight", default=1)]
            )
        min_color = min(
            [d for (u, v, d) in G.edges(data="weight", default=1)])
        # color range: gray to black
        edge_vmin = 2 * min_color - edge_vmax

    nx.draw(
        G,
        pos=pos_layout,
        arrows=True,
        with_labels=True,
        font_size=4,
        labels=feat_labels,
        node_color=node_colors,
        vmin=0,
        vmax=vmax,
        cmap=cmap,
        edge_color=edge_colors,
        edge_cmap=plt.get_cmap("Greys"),
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        width=1.0,
        node_size=120,
        alpha=0.8,
    )
    fig.axes[0].xaxis.set_visible(False)
    fig.canvas.draw()


def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
            node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
                    node(s).
            num_hops: (int): The number of hops :math:`k`.
            edge_index (LongTensor): The edge indices.
            relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
                    :obj:`edge_index` will be relabeled to hold consecutive indices
                    starting from zero. (default: :obj:`False`)
            num_nodes (int, optional): The number of nodes, *i.e.*
                    :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
            flow (string, optional): The flow direction of :math:`k`-hop
                    aggregation (:obj:`"source_to_target"` or
                    :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
                     :class:`BoolTensor`)
    """

    def maybe_num_nodes(index, num_nodes=None):
        return index.max().item() + 1 if num_nodes is None else num_nodes

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


def custom_to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                       remove_self_loops=False):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.DiGraph` if :attr:`to_undirected` is set to :obj:`True`, or
    an undirected :obj:`networkx.Graph` otherwise.
    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)
    """

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(range(data.num_nodes))

    values = {}
    for key, item in data.__dict__.items():
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)
        for key in edge_attrs if edge_attrs is not None else []:
            G[u][v][key] = values[key][i]

    for key in node_attrs if node_attrs is not None else []:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G





































class BaseExplainer(ABC):
    def __init__(self, model_to_explain, graphs, features, task):
        self.model_to_explain = model_to_explain
        self.graphs = graphs
        self.features = features
        self.type = task

    @abstractmethod
    def prepare(self, args):
        """Prepars the explanation method for explaining.
        Can for example be used to train the method"""
        pass

    @abstractmethod
    def explain(self, index):
        """
        Main method for explaining samples
        :param index: index of node/graph in self.graphs
        :return: explanation for sample
        """
        pass




class GraphSVX(BaseExplainer):
    def __init__(self, model_to_explain, graphs, features, task, gpu=False, **kwargs):
        super().__init__(model_to_explain, graphs, features, task)

        self.data = None#data

        self.gpu = gpu
        self.neighbours = None  # nodes considered
        self.F = None  # number of features considered
        self.M = None  # number of features and nodes considered
        self.base_values = []

        #self.model_to_explain.eval()

    ################################
    # Core function - explain
    ################################

    def prepare(self, args):
        """Nothing is done to prepare the GNNExplainer, this happens at every index"""
        return


    def explain_node(self,
                node_index,
                hops=2,
                num_samples=10,
                info=True,
                multiclass=False,
                fullempty=None,
                S=3,
                args_hv='compute_pred',
                args_feat='Expectation',
                args_coal='Smarter',
                args_g='WLS',
                regu=None,
                vizu=False):


        # Compute true prediction for original instance via explained GNN model
        if self.gpu:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            #self.model_to_explain = self.model_to_explain.to(device)
            with torch.no_grad():
                true_conf, true_pred = self.model_to_explain(
                    self.features.cuda(),
                    self.graphs.cuda()).exp()[node_index].max(dim=0)
        else:
            with torch.no_grad():
                true_conf, true_pred = self.model_to_explain(
                    self.features,
                    self.graphs).exp()[node_index].max(dim=0)

        # --- Node selection ---
        # Investigate k-hop subgraph of the node of interest (v)
        self.neighbours, _, _, edge_mask = \
            torch_geometric.utils.k_hop_subgraph(node_idx=node_index,
                                                 num_hops=hops,
                                                 edge_index=self.graphs)

        # Retrieve 1-hop neighbours of v
        one_hop_neighbours, _, _, _ = \
            torch_geometric.utils.k_hop_subgraph(node_idx=node_index,
                                                 num_hops=1,
                                                 edge_index=self.graphs)
        # Stores the indexes of the neighbours of v (+ index of v itself)

        # Remove node v index from neighbours and store their number in D
        self.neighbours = self.neighbours[self.neighbours != node_index]
        D = self.neighbours.shape[0]

        # --- Feature selection ---
        if args_hv == 'compute_pred_subgraph':
            feat_idx, discarded_feat_idx = self.feature_selection_subgraph(node_index, args_feat)
            # Also solve incompatibility due to overlap feat/node importance
            if args_hv == 'SmarterSeparate' or args_hv == 'NewSmarterSeparate':
                print('Incompatibility: user Smarter sampling instead')
                args_hv = 'Smarter'
        else:
            feat_idx, discarded_feat_idx = self.feature_selection(node_index, args_feat)

        # M: total number of features + neighbours considered for node v
        if regu == 1 or D == 0:
            D = 0
            print('Explainations only consider node features')
        if regu == 0 or self.F == 0:
            self.F = 0
            print('Explainations only consider graph structure')
        self.M = self.F + D

        # Def range of endcases considered
        args_K = S

        # --- MASK GENERATOR ---
        # Generate binary samples z' representing coalitions of nodes and features
        z_, weights = self.mask_generation(num_samples, args_coal, args_K, D, info, regu)

        # Discard full and empty coalition if specified
        if fullempty:
            weights[weights == 1000] = 0

        # --- GRAPH GENERATOR ---
        # Create dataset (z, f(GEN(z'))), stored as (z_, fz)
        # Retrieve z' from z and x_v, then compute f(z')
        fz = eval('self.' + args_hv)(node_index, num_samples, D, z_,
                                     feat_idx, one_hop_neighbours, args_K, args_feat,
                                     discarded_feat_idx, multiclass, true_pred)

        # --- EXPLANATION GENERATOR ---
        # Train Surrogate Weighted Linear Regression - learns shapley values
        phi, base_value = eval('self.' + args_g)(z_,
                                                 weights, fz, multiclass, info)

        # Rescale
        if type(regu) == int and not multiclass:
            expl = (true_conf.cpu() - base_value).detach().numpy()
            phi[:self.F] = (regu * expl / sum(phi[:self.F])) * phi[:self.F]
            phi[self.F:] = ((1 - regu) * expl /
                            sum(phi[self.F:])) * phi[self.F:]

        # Print information
        if info:
            self.print_info(D, node_index, phi, feat_idx,
                            true_pred, true_conf, base_value, multiclass)

        # Visualise
        if vizu:
            self.vizu(edge_mask, node_index, phi,
                      true_pred, hops, multiclass)

        # Time
        end = time.time()
        if info:
            print('Time: ', end - start)

        # Append explanations for this node to list of expl.


        self.base_values.append(base_value)
        return phi
    def explain_instances(self,
                node_indexes=[0],
                hops=2,
                num_samples=10,
                info=True,
                multiclass=False,
                fullempty=None,
                S=3,
                args_hv='compute_pred',
                args_feat='Expectation',
                args_coal='Smarter',
                args_g='WLS',
                regu=None,
                vizu=False):
        """ Explain prediction for a given node - GraphSVX method
        Args:
            node_indexes (list, optional): indexes of the nodes of interest. Defaults to [0].
            hops (int, optional): number k of k-hop neighbours to consider in the subgraph
                                                    around node_index. Defaults to 2.
            num_samples (int, optional): number of samples we want to form GraphSVX's new dataset.
                                                    Defaults to 10.
            info (bool, optional): Print information about explainer's inner workings.
                                                    And include vizualisation. Defaults to True.
            multiclass (bool, optional): extension - consider predicted class only or all classes
            fullempty (bool, optional): enforce high weight for full and empty coalitions
            S (int, optional): maximum size of coalitions that are favoured in mask generation phase
            args_hv (str, optional): strategy used to convert simplified input z to original
                                                    input space z'
            args_feat (str, optional): way to switch off and discard node features (0 or expectation)
            args_coal (str, optional): how we sample coalitions z
            args_g (str, optional): method used to train model g on (z, f(z'))
            regu (int, optional): extension - apply regularisation to balance importance granted
                                                    to nodes vs features
            vizu (bool, optional): creates vizualisation or not
        Returns:
                [tensors]: shapley values for features/neighbours that influence node v's pred
                        and base value
        """
        # Time
        start = time.time()

        # Explain several nodes sequentially
        phi_list = []
        for node_index in node_indexes:
            phi_list.append(self.explain_node(node_index,
                hops,
                num_samples,
                info,
                multiclass,
                fullempty,
                S,
                args_hv,
                args_feat,
                args_coal,
                args_g,
                regu,
                vizu))
        return phi_list

    def explain(self,
                graph_index,
                hops=2,
                num_samples=10,
                info=False,
                multiclass=False,
                fullempty=None,
                S=4,
                args_hv='compute_pred',
                args_feat='All',
                args_coal='SmarterSeparate',
                args_g='WLR_sklearn',
                regu=None,
                vizu=False
                ):

        """ Explains prediction for a graph classification task - GraphSVX method
        Args:
            graph_index (list, optional): indexes of the graph of interest. Defaults to [0].
            hops (int, optional): number k of k-hop neighbours to consider in the subgraph
                                                    around node_index. Defaults to 2.
            num_samples (int, optional): number of samples we want to form GraphSVX's new dataset.
                                                    Defaults to 10.
            info (bool, optional): Print information about explainer's inner workings.
                                                    And include vizualisation. Defaults to True.
            multiclass (bool, optional): extension - consider predicted class only or all classes
            fullempty (bool, optional): enforce high weight for full and empty coalitions
            S (int, optional): maximum size of coalitions that are favoured in mask generation phase
            args_hv (str, optional): strategy used to convert simplified input z to original
                                                    input space z'
            args_feat (str, optional): way to switch off and discard node features (0 or expectation)
            args_coal (str, optional): how we sample coalitions z
            args_g (str, optional): method used to train model g on (z, f(z'))
            regu (int, optional): extension - apply regularisation to balance importance granted
                                                    to nodes vs features
            vizu (bool, optional): creates vizualisation or not
        Returns:
                [tensors]: shapley values for features/neighbours that influence node v's pred
                        and base value
        """

        # Compute true prediction for original instance via explained GNN model
        features= self.features[graph_index]
        graph = self.graphs[graph_index]
        if self.gpu:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.model_to_explain = self.model_to_explain.to(device)
            with torch.no_grad():
                true_conf, true_pred = self.model_to_explain(features.cuda(),
                                                             graph.cuda()).exp()[graph_index, :].max(dim=0)
        else:
            with torch.no_grad():
                true_conf, true_pred = self.model_to_explain(features,
                                                             graph).max(dim=1)

        # Remove node v index from neighbours and store their number in D
        #self.neighbours = list(
        #    range(int(graph.shape[1] -features.shape[0])))
        #D = len(self.neighbours)
        D = int(features.sum().item())

        nx_graph= networkx.from_edgelist(graph.numpy().T)

        self.neighbours = [list(nx_graph.adj[el]) for el in list(nx_graph.nodes)[:D]]

        #self.neighbours= list(range(D))
        # Total number of features + neighbours considered for node v
        self.F = 0#features.shape[1]
        self.M = self.F + D

        # Def range of endcases considered
        args_K = S

        # --- MASK GENERATOR ---
        z_, weights = self.mask_generation(num_samples, args_coal, args_K, D, info, regu)

        # Discard full and empty coalition if specified
        if fullempty:
            weights[(weights == 1000).nonzero()] = 0

        # --- GRAPH GENERATOR ---
        # Create dataset (z, f(GEN(z'))), stored as (z_, fz)
        # Retrieve z' from z and x_v, then compute f(z')
        fz = self.graph_classification(
            graph_index, num_samples, D, z_, args_K, args_feat, true_pred)

        # --- EXPLANATION GENERATOR ---
        # Train Surrogate Weighted Linear Regression - learns shapley values
        phi, base_value = eval('self.' + args_g)(z_, weights, fz,
                                                 multiclass, info)

        self.base_values.append(base_value)
        phi = phi - phi.min()
        if (graph.shape[1]-D)<0:
            return self.graphs[graph_index].detach(), torch.zeros(self.graphs[graph_index].shape[1])
        phi_out = torch.cat([torch.tensor(phi,dtype=torch.float32), torch.zeros(int(graph.shape[1]-D))])

        return self.graphs[graph_index].detach(), phi_out
        #torch.cat(phi, np.zeros(int(features.shape[0]-D)))

    def explain_graphs(self,
                       graph_indices=[0],
                       hops=2,
                       num_samples=100,
                       info=True,
                       multiclass=False,
                       fullempty=None,
                       S=3,
                       args_hv='compute_pred',
                       args_feat='Expectation',
                       args_coal='Smarter',
                       args_g='WLS',
                       regu=None,
                       vizu=False):
        """ Explains prediction for a graph classification task - GraphSVX method
        Args:
            node_indexes (list, optional): indexes of the nodes of interest. Defaults to [0].
            hops (int, optional): number k of k-hop neighbours to consider in the subgraph
                                                    around node_index. Defaults to 2.
            num_samples (int, optional): number of samples we want to form GraphSVX's new dataset.
                                                    Defaults to 10.
            info (bool, optional): Print information about explainer's inner workings.
                                                    And include vizualisation. Defaults to True.
            multiclass (bool, optional): extension - consider predicted class only or all classes
            fullempty (bool, optional): enforce high weight for full and empty coalitions
            S (int, optional): maximum size of coalitions that are favoured in mask generation phase
            args_hv (str, optional): strategy used to convert simplified input z to original
                                                    input space z'
            args_feat (str, optional): way to switch off and discard node features (0 or expectation)
            args_coal (str, optional): how we sample coalitions z
            args_g (str, optional): method used to train model g on (z, f(z'))
            regu (int, optional): extension - apply regularisation to balance importance granted
                                                    to nodes vs features
            vizu (bool, optional): creates vizualisation or not
        Returns:
                [tensors]: shapley values for features/neighbours that influence node v's pred
                        and base value
        """

        # Time
        start = time.time()

        # --- Explain several nodes iteratively ---
        phi_list = []
        for graph_index in graph_indices:
            phi_list.append(self.explain(graph_index,
                                         hops,
                                         num_samples,
                                         info,
                                         multiclass,
                                         fullempty,
                                         S,
                                         args_hv,
                                         args_feat,
                                         args_coal,
                                         args_g,
                                         regu,
                                         vizu
                                         ))

            return phi_list

    ################################
    # Feature selector
    ################################

    def feature_selection(self, node_index, args_feat):
        """ Select features who truly impact prediction
        Others will receive a 0 shapley value anyway
        Args:
            node_index (int): node index
            args_feat (str): strategy utilised to select
                                important featutres
        Returns:
            [tensor]: list of important features' index
            [tensor]: list of discarded features' index
        """

        # Only consider relevant features in explanations
        discarded_feat_idx = []
        if args_feat == 'All':
            # Select all features
            self.F = self.features[node_index, :].shape[0]
            feat_idx = torch.unsqueeze(
                torch.arange(self.data.num_nodes), 1)
        elif args_feat == 'Null':
            # Select features whose value is non-null
            feat_idx = self.features[node_index, :].nonzero()
            self.F = feat_idx.size()[0]
        else:
            # Select features whose value is different from dataset mean value
            std = self.features.std(axis=0)
            mean = self.features.mean(axis=0)
            mean_subgraph = self.features[node_index, :]
            mean_subgraph = torch.where(mean_subgraph >= mean - 0.25 * std, mean_subgraph,
                                        torch.ones_like(mean_subgraph) * 100)
            mean_subgraph = torch.where(mean_subgraph <= mean + 0.25 * std, mean_subgraph,
                                        torch.ones_like(mean_subgraph) * 100)
            feat_idx = (mean_subgraph == 100).nonzero()
            discarded_feat_idx = (mean_subgraph != 100).nonzero()
            self.F = feat_idx.shape[0]

        return feat_idx, discarded_feat_idx

    def feature_selection_subgraph(self, node_index, args_feat):
        """ Similar to feature_selection (above)
        but considers the feature vector in the subgraph around v
        instead of the feature of v
        """
        # Specific case: features in subgraph
        # Determine features and neighbours whose importance is investigated
        discarded_feat_idx = []
        if args_feat == 'All':
            # Consider all features - no selection
            self.F = self.features[node_index, :].shape[0]
            feat_idx = torch.unsqueeze(
                torch.arange(self.data.num_nodes), 1)
        elif args_feat == 'Null':
            # Consider only non-zero entries in the subgraph of v
            feat_idx = self.features[self.neighbours, :].mean(
                axis=0).nonzero()
            self.F = feat_idx.size()[0]
        else:
            # Consider all features away from its mean value
            std = self.features.std(axis=0)
            mean = self.features.mean(axis=0)
            # Feature intermediate rep
            mean_subgraph = torch.mean(self.features[self.neighbours, :], dim=0)
            # Select relevant features only - (E-e,E+e)
            mean_subgraph = torch.where(mean_subgraph >= mean - 0.25 * std, mean_subgraph,
                                        torch.ones_like(mean_subgraph) * 100)
            mean_subgraph = torch.where(mean_subgraph <= mean + 0.25 * std, mean_subgraph,
                                        torch.ones_like(mean_subgraph) * 100)
            feat_idx = (mean_subgraph == 100).nonzero()
            discarded_feat_idx = (mean_subgraph != 100).nonzero()
            self.F = feat_idx.shape[0]
            del mean, mean_subgraph, std

        return feat_idx, discarded_feat_idx

    ################################
    # Mask generator
    ################################

    def mask_generation(self, num_samples, args_coal, args_K, D, info, regu):
        """ Applies selected mask generator strategy
        Args:
            num_samples (int): number of samples for GraphSVX
            args_coal (str): mask generator strategy
            args_K (int): size param for indirect effect
            D (int): number of nodes considered after selection
            info (bool): print information or not
            regu (int): balances importance granted to nodes and features
        Returns:
            [tensor] (num_samples, M): dataset of samples/coalitions z'
            [tensor] (num_samples): vector of kernel weights corresponding to samples
        """
        if args_coal == 'SmarterSeparate' or args_coal == 'NewSmarterSeparate':
            weights = torch.zeros(num_samples, dtype=torch.float64)
            if self.F == 0 or D == 0:
                num = int(num_samples * self.F / self.M)
            elif regu != None:
                num = int(num_samples * regu)
                # num = int( num_samples * ( self.F/self.M + ((regu - 0.5)/0.5)  * (self.F/self.M) ) )
            else:
                num = int(0.5 * num_samples / 2 + 0.5 * num_samples * self.F / self.M)
            # Features only
            z_bis = eval('self.' + args_coal)(num, args_K, 1)
            z_bis = z_bis[torch.randperm(z_bis.size()[0])]
            s = (z_bis != 0).sum(dim=1)
            weights[:num] = self.shapley_kernel(s, self.F)
            z_ = torch.zeros(num_samples, self.M)
            z_[:num, :self.F] = z_bis
            # Node only
            z_bis = eval('self.' + args_coal)(
                num_samples - num, args_K, 0)
            z_bis = z_bis[torch.randperm(z_bis.size()[0])]
            s = (z_bis != 0).sum(dim=1)
            weights[num:] = self.shapley_kernel(s, D)
            z_[num:, :] = torch.ones(num_samples - num, self.M)
            z_[num:, self.F:] = z_bis

        else:
            # If we choose to sample all possible coalitions
            if args_coal == 'All':
                num_samples = min(10000, 2 ** self.M)

            # Coalitions: sample num_samples binary vectors of dimension M
            z_ = eval('self.' + args_coal)(num_samples, args_K, regu)

            # Shuffle them
            z_ = z_[torch.randperm(z_.size()[0])]

            # Compute |z| for each sample z: number of non-zero entries
            s = (z_ != 0).sum(dim=1)

            # GraphSVX Kernel: define weights associated with each sample
            weights = self.shapley_kernel(s, self.M)

        return z_, weights

    def NewSmarterSeparate(self, num_samples, args_K, regu):
        """Default mask sampler
        Generates feature mask and node mask independently
        Favours masks with a high weight + smart space allocation algorithm
        Args:
            num_samples (int): number of masks desired
            args_K (int): maximum size of masks favoured
            regu (binary): nodes or features
        Returns:
            tensor: dataset of samples
        """
        if regu == None:
            z_ = self.Smarter(num_samples, args_K, regu)
            return z_

        # Favour features - special coalitions don't study node's effect
        elif regu > 0.5:
            M = self.F
            z_ = torch.ones(num_samples, M)
            z_[1::2] = torch.zeros(num_samples // 2, M)  # case k = 0
            i = 2
            k = 1
            P = num_samples * 9 / 10
            while i < P and k <= min(args_K, M - 1):

                # All coalitions of order k can be sampled
                if i + 2 * scipy.special.comb(M, k) <= P:
                    L = list(combinations(range(M), k))
                    for c in L:
                        z_[i, c] = torch.zeros(k)
                        z_[i + 1, c] = torch.ones(k)
                        i += 2

                # All coalitions of order k cannot be sampled
                else:
                    weight = torch.ones(M)
                    L = list(combinations(range(M), k))
                    random.shuffle(L)
                    while i < min(P, len(L)):
                        cw = torch.tensor([sum(weight[list(c)]) for c in L])
                        c_idx = torch.argmax(cw).item()
                        c = list(L[c_idx])
                        p = float(random.randint(0, 1))
                        z_[i, :] = torch.tensor(p).repeat(M)
                        z_[i, c] = torch.tensor(1 - p).repeat(len(c))
                        weight[list(c)] = torch.tensor([1 / (1 + 1 / el.item()) for el in weight[list(c)]])
                        i += 1
                    k += 1

            # Random coal
            z_[i:, :] = torch.empty(max(0, num_samples - i), M).random_(2)
            return z_

        # Favour features - special coalitions don't study node's effect
        elif regu < 0.5:
            M = self.M - self.F
            z_ = torch.ones(num_samples, M)
            z_[1::2] = torch.zeros(num_samples // 2, M)  # case k = 0
            i = 2
            k = 1
            P = int(num_samples * 9 / 10)
            while i < P and k <= min(args_K, M - 1):

                # All coalitions of order k can be sampled
                if i + 2 * scipy.special.comb(M, k) <= P:
                    L = list(combinations(range(M), k))
                    for c in L:
                        z_[i, c] = torch.zeros(k)
                        z_[i + 1, c] = torch.ones(k)
                        i += 2
                    k += 1

                # All coalitions of order k cannot be sampled
                else:
                    weight = torch.ones(M)
                    L = list(combinations(range(M), k))
                    random.shuffle(L)
                    while i < min(P, len(L)):
                        cw = torch.tensor([sum(weight[list(c)]) for c in L])
                        c_idx = torch.argmax(cw).item()
                        c = list(L[c_idx])
                        p = float(random.randint(0, 1))
                        z_[i, :] = torch.tensor(p).repeat(M)
                        z_[i, c] = torch.tensor(1 - p).repeat(len(c))
                        weight[list(c)] = torch.tensor([1 / (1 + 1 / el.item()) for el in weight[list(c)]])
                        i += 1
                    k += 1

            # Random coal
            z_[i:, :] = torch.empty(max(0, num_samples - i), M).random_(2)
            return z_

    def SmarterSeparate(self, num_samples, args_K, regu):
        """Default mask sampler
        Generates feature mask and node mask independently
        Favours masks with a high weight + efficient space allocation algorithm
        Args:
            num_samples (int): number of masks desired
            args_K (int): maximum size of masks favoured
            regu (binary): nodes or features
        Returns:
            tensor: dataset of samples
        """
        if regu == None:
            z_ = self.Smarter(num_samples, args_K, regu)
            return z_

        # Favour features - special coalitions don't study node's effect
        elif regu > 0.5:
            # Define empty and full coalitions
            M = self.F
            z_ = torch.ones(num_samples, M)
            z_[1::2] = torch.zeros(num_samples // 2, M)
            # z_[1, :] = torch.empty(1, self.M).random_(2)
            i = 2
            k = 1
            # Loop until all samples are created
            while i < num_samples:
                # Look at each feat/nei individually if have enough sample
                # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
                if i + 2 * M < num_samples and k == 1:
                    z_[i:i + M, :] = torch.ones(M, M)
                    z_[i:i + M, :].fill_diagonal_(0)
                    z_[i + M:i + 2 * M, :] = torch.zeros(M, M)
                    z_[i + M:i + 2 * M, :].fill_diagonal_(1)
                    i += 2 * M
                    k += 1

                else:
                    # Split in two number of remaining samples
                    # Half for specific coalitions with low k and rest random samples
                    samp = i + 9 * (num_samples - i) // 10
                    # samp = num_samples
                    while i < samp and k <= min(args_K, M):
                        # Sample coalitions of k1 neighbours or k1 features without repet and order.
                        L = list(combinations(range(M), k))
                        random.shuffle(L)
                        L = L[:samp + 1]

                        for j in range(len(L)):
                            # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                            z_[i, L[j]] = torch.zeros(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                return z_
                            # Coalitions (No nei, k feat) or (No feat, k nei)
                            z_[i, L[j]] = torch.ones(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                return z_
                        k += 1

                    # Sample random coalitions
                    z_[i:, :] = torch.empty(num_samples - i, M).random_(2)
                    return z_
            return z_

        # Favour neighbour
        else:
            # Define empty and full coalitions
            M = self.M - self.F
            # self.F = 0
            z_ = torch.ones(num_samples, M)
            z_[1::2] = torch.zeros(num_samples // 2, M)
            i = 2
            k = 1
            # Loop until all samples are created
            while i < num_samples:
                # Look at each feat/nei individually if have enough sample
                # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
                if i + 2 * M < num_samples and k == 1:
                    z_[i:i + M, :] = torch.ones(M, M)
                    z_[i:i + M, :].fill_diagonal_(0)
                    z_[i + M:i + 2 * M, :] = torch.zeros(M, M)
                    z_[i + M:i + 2 * M, :].fill_diagonal_(1)
                    i += 2 * M
                    k += 1

                else:
                    # Split in two number of remaining samples
                    # Half for specific coalitions with low k and rest random samples
                    # samp = i + 9*(num_samples - i)//10
                    samp = num_samples
                    while i < samp and k <= min(args_K, M):
                        # Sample coalitions of k1 neighbours or k1 features without repet and order.
                        L = list(combinations(range(0, M), k))
                        random.shuffle(L)
                        L = L[:samp + 1]

                        for j in range(len(L)):
                            # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                            z_[i, L[j]] = torch.zeros(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                z_[i:, :] = torch.empty(num_samples - i, M).random_(2)
                                return z_
                            # Coalitions (No nei, k feat) or (No feat, k nei)
                            z_[i, L[j]] = torch.ones(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                z_[i:, :] = torch.empty(num_samples - i, M).random_(2)
                                return z_
                        k += 1

                    # Sample random coalitions
                    z_[i:, :] = torch.empty(num_samples - i, M).random_(2)
                    return z_
            return z_

    def Smarter(self, num_samples, args_K, *unused):
        """ Smart Mask generator
        Nodes and features are considered together but separately
        Args:
            num_samples ([int]): total number of coalitions z_
            args_K: max size of coalitions favoured in sampling
        Returns:
            [tensor]: z_ in {0,1}^F x {0,1}^D (num_samples x self.M)
        """
        # Define empty and full coalitions
        z_ = torch.ones(num_samples, self.M)
        z_[1::2] = torch.zeros(num_samples // 2, self.M)
        i = 2
        k = 1
        # Loop until all samples are created
        while i < num_samples:
            # Look at each feat/nei individually if have enough sample
            # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
            if i + 2 * self.M < num_samples and k == 1:
                z_[i:i + self.M, :] = torch.ones(self.M, self.M)
                z_[i:i + self.M, :].fill_diagonal_(0)
                z_[i + self.M:i + 2 * self.M, :] = torch.zeros(self.M, self.M)
                z_[i + self.M:i + 2 * self.M, :].fill_diagonal_(1)
                i += 2 * self.M
                k += 1

            else:
                # Split in two number of remaining samples
                # Half for specific coalitions with low k and rest random samples
                samp = i + 9 * (num_samples - i) // 10
                while i < samp and k <= args_K:
                    # Sample coalitions of k1 neighbours or k1 features without repet and order.
                    L = list(combinations(range(self.F), k)) + \
                        list(combinations(range(self.F, self.M), k))
                    random.shuffle(L)
                    L = L[:samp + 1]

                    for j in range(len(L)):
                        # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                        z_[i, L[j]] = torch.zeros(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(
                                num_samples - i, self.M).random_(2)
                            return z_
                        # Coalitions (No nei, k feat) or (No feat, k nei)
                        z_[i, L[j]] = torch.ones(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(
                                num_samples - i, self.M).random_(2)
                            return z_
                    k += 1

                # Sample random coalitions
                z_[i:, :] = torch.empty(num_samples - i, self.M).random_(2)
                return z_
        return z_

    def Smart(self, num_samples, args_K, *unused):
        """ Sample coalitions cleverly
        Favour coalition with height weight - no distinction nodes/feat
        Args:
            num_samples (int): total number of coalitions z_
            args_K (int): max size of coalitions favoured
        Returns:
            [tensor]: z_ in {0,1}^F x {0,1}^D (num_samples x self.M)
        """
        z_ = torch.ones(num_samples, self.M)
        z_[1::2] = torch.zeros(num_samples // 2, self.M)
        k = 1
        i = 2
        while i < num_samples:
            if i + 2 * self.M < num_samples and k == 1:
                z_[i:i + self.M, :] = torch.ones(self.M, self.M)
                z_[i:i + self.M, :].fill_diagonal_(0)
                z_[i + self.M:i + 2 * self.M, :] = torch.zeros(self.M, self.M)
                z_[i + self.M:i + 2 * self.M, :].fill_diagonal_(1)
                i += 2 * self.M
                k += 1
            elif k == 1:
                M = list(range(self.M))
                random.shuffle(M)
                for j in range(self.M):
                    z_[i, M[j]] = torch.zeros(1)
                    i += 1
                    if i == num_samples:
                        return z_
                    z_[i, M[j]] = torch.ones(1)
                    i += 1
                    if i == num_samples:
                        return z_
                k += 1
            elif k < args_K:
                samp = i + 4 * (num_samples - i) // 5
                M = list(combinations(range(self.M), k))[:samp - i + 1]
                random.shuffle(M)
                for j in range(len(M)):
                    z_[i, M[j][0]] = torch.tensor(0)
                    z_[i, M[j][1]] = torch.tensor(0)
                    i += 1
                    if i == samp:
                        z_[i:, :] = torch.empty(
                            num_samples - i, self.M).random_(2)
                        return z_
                    z_[i, M[j][0]] = torch.tensor(1)
                    z_[i, M[j][1]] = torch.tensor(1)
                    i += 1
                    if i == samp:
                        z_[i:, :] = torch.empty(
                            num_samples - i, self.M).random_(2)
                        return z_
                k += 1
            else:
                z_[i:, :] = torch.empty(num_samples - i, self.M).random_(2)
                return z_

        return z_

    def Random(self, num_samples, *unused):
        """Sample masks randomly
        """
        z_ = torch.empty(num_samples, self.M).random_(2)
        return z_

    def All(self, num_samples, *unsused):
        """Sample all possible 2^{F+N} coalitions (unordered, without replacement)
        Args:
            num_samples (int): 2^{M+N} or boundary we fixed (20,000)
        [tensor]: dataset (2^{M+N} x self.M) where each row is in {0,1}^F x {0,1}^D
        """
        z_ = torch.zeros(num_samples, self.M)
        i = 0
        try:
            for k in range(0, self.M + 1):
                L = list(combinations(range(0, self.M), k))
                for j in range(len(L)):
                    z_[i, L[j]] = torch.ones(k)
                    i += 1
        except IndexError:  # deal with boundary
            return z_
        return z_

    ################################
    # GraphSVX kernel
    ################################

    def shapley_kernel(self, s, M):
        """ Computes a weight for each newly created sample
        Args:
            s (tensor): contains dimension of z for all instances
                (number of features + neighbours included)
            M (tensor): total number of features/nodes in dataset
        Returns:
                [tensor]: shapley kernel value for each sample
        """
        shapley_kernel = []

        for i in range(s.shape[0]):
            a = s[i].item()
            if a == 0 or a == M:
                # Enforce high weight on full/empty coalitions
                shapley_kernel.append(1000)
            elif scipy.special.binom(M, a) == float('+inf'):
                # Treat specific case - impossible computation
                shapley_kernel.append(1 / (M ** 2))
            else:
                shapley_kernel.append(
                    (M - 1) / (scipy.special.binom(M, a) * a * (M - a)))

        shapley_kernel = np.array(shapley_kernel)
        shapley_kernel = np.where(shapley_kernel < 1.0e-40, 1.0e-40, shapley_kernel)
        return torch.tensor(shapley_kernel)

    ################################
    # Graph generator + compute f(z')
    ################################

    def compute_pred_subgraph(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat,
                              discarded_feat_idx, multiclass, true_pred):
        """ Construct z' from z and compute prediction f(z') for each sample z
            In fact, we build the dataset (z, f(z')), required to train the weighted linear model.
            Features in subgraph
        Args:
                Variables are defined exactly as defined in explainer function
        Returns:
                (tensor): f(z') - probability of belonging to each target classes, for all samples z'
                Dimension (N * C) where N is num_samples and C num_classses.
        """
        # Create networkx graph
        G = custom_to_networkx(self.data)
        G = G.subgraph(self.neighbours.tolist() + [node_index])

        # Define an "average" feature vector - for discarded features
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.num_features)
        else:
            av_feat_values = self.features.mean(dim=0)
            # Change here for contrastive explanations
            # av_feat_values = self.features[402]
            # or random feature vector made of random value across each col of X

        # Init dict for nodes and features not sampled
        excluded_feat = {}
        excluded_nei = {}

        # Define excluded_feat and excluded_nei for each z
        for i in range(num_samples):

            # Store index of features that are not sampled (z_j=0)
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Store index of neighbours that need to be isolated (not sampled, z_j=0)
            nodes_id = []
            for j in range(D):
                if z_[i, self.F + j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            # Dico with key = num_sample id, value = excluded neighbour index
            excluded_nei[i] = nodes_id

        # Init label f(z') for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)
        # classes_labels = torch.zeros(num_samples)
        # pred_confidence = torch.zeros(num_samples)

        # Create new matrix A and X - for each sample ≈ reform z from z
        for (key, ex_nei), (_, ex_feat) in tqdm(zip(excluded_nei.items(), excluded_feat.items())):

            # For each excluded neighbour, retrieve the column index of its occurences
            # in the adj matrix - store them in positions (list)
            positions = []
            for val in ex_nei:
                pos = (self.graphs == val).nonzero()[:, 1].tolist()
                positions += pos
            positions = list(set(positions))
            A = np.array(self.graphs)
            # Special case (0 node, k feat)
            # Consider only feat. influence if too few nei included
            if D - len(ex_nei) >= min(self.F - len(ex_feat), args_K):
                A = np.delete(A, positions, axis=1)
            A = torch.tensor(A)

            # Change feature vector for node of interest - excluded and discarded features
            X = deepcopy(self.features)
            X[node_index, ex_feat] = av_feat_values[ex_feat]
            if args_feat != 'Null' and discarded_feat_idx != [] and len(self.neighbours) - len(ex_nei) < args_K:
                X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]

                # End approximation
                # if args_feat == 'Expectation':
                #    for val in discarded_feat_idx:
                #        X[self.neighbours, val] = av_feat_values[val].repeat(D)

            # Special case - consider only nei. influence if too few feat included
            if self.F - len(ex_feat) < min(D - len(ex_nei), args_K):
                # Don't set features = Exp or 0 in the whole subgraph, only for v.

                # Indirect effect
                included_nei = set(
                    self.neighbours.detach().numpy()).difference(ex_nei)
                included_nei = included_nei.difference(
                    one_hop_neighbours.detach().numpy())
                for incl_nei in included_nei:
                    paths = list(nx.all_shortest_paths(G, source=node_index, target=incl_nei))
                    np.random.shuffle(paths)
                    len_paths = [len(set(path[1:-1]).intersection(ex_nei)) for path in paths]
                    if min(len_paths) == 0:
                        pass
                    else:
                        path = paths[np.argmin(len_paths)]
                        for n in range(1, len(path) - 1):
                            A = torch.cat((A, torch.tensor(
                                [[path[n - 1]], [path[n]]])), dim=-1)
                            X[path[n], :] = av_feat_values

            # Usual case - exclude features for the whole subgraph
            else:
                for val in ex_feat:
                    X[self.neighbours, val] = av_feat_values[val].repeat(len(self.neighbours))

            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    proba = self.model_to_explain(X.cuda(), A.cuda()).exp()[node_index]
            else:
                with torch.no_grad():
                    proba = self.model_to_explain(X, A).exp()[node_index]

            # Store final class prediction and confience level
            # pred_confidence[key], classes_labels[key] = torch.topk(proba, k=1)

            # Store predicted class label in fz
            if multiclass:
                fz[key] = proba
            else:
                fz[key] = proba[true_pred]

        return fz

    def compute_pred(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat,
                     discarded_feat_idx, multiclass, true_pred):
        """ Construct z' from z and compute prediction f(z') for each sample z
            In fact, we build the dataset (z, f(z')), required to train the weighted linear model.
            Standard method
        Args:
                Variables are defined exactly as defined in explainer function
        Returns:
                (tensor): f(z') - probability of belonging to each target classes, for all samples z'
                Dimension (N * C) where N is num_samples and C num_classses.
        """
        # Create a networkx graph
        G = custom_to_networkx(self.data)
        G = G.subgraph(self.neighbours.tolist() + [node_index])

        # Define an "average" feature vector - for discarded features
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.num_features)
        else:
            av_feat_values = self.features.mean(dim=0)
            # Change here for contrastive explanations
            # av_feat_values = self.features[402]
            # or random feature vector made of random value across each col of X

        # Store discarded nodes/features (z_j=0) for each sample z
        excluded_feat = {}
        excluded_nei = {}
        for i in range(num_samples):

            # Excluded features' indexes
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Excluded neighbors' indexes
            nodes_id = []
            for j in range(D):
                if z_[i, self.F + j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            excluded_nei[i] = nodes_id
            # Dico with key = num_sample id, value = excluded neighbour index

        # Init label f(z') for graphshap dataset
        if multiclass:
            # Allows to explain why one class was not chosen
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        # Construct new matrices A and X for each sample - reform z' from z
        for (key, ex_nei), (_, ex_feat) in tqdm(zip(excluded_nei.items(), excluded_feat.items())):

            # Isolate in the graph each node excluded from the sampled coalition
            positions = []
            for val in ex_nei:
                pos = (self.graphs == val).nonzero()[:, 1].tolist()
                positions += pos
            positions = list(set(positions))
            A = np.array(self.graphs)
            A = np.delete(A, positions, axis=1)
            A = torch.tensor(A)

            # Set features not in the sampled coalition to an average value
            X = deepcopy(self.features)
            X[node_index, ex_feat] = av_feat_values[ex_feat]

            # Discared features approximation
            # if args_feat != 'Null' and discarded_feat_idx!=[] and D - len(ex_nei) < args_K:
            #     X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]

            # Indirect effect - if few included neighbours
            # Make sure that they are connected to v (with current nodes sampled nodes)
            if 0 < D - len(ex_nei) < args_K:
                included_nei = set(
                    self.neighbours.detach().numpy()).difference(ex_nei)
                included_nei = included_nei.difference(
                    one_hop_neighbours.detach().numpy())
                for incl_nei in included_nei:
                    paths = list(nx.all_shortest_paths(G, source=node_index, target=incl_nei))
                    np.random.shuffle(paths)
                    len_paths = [len(set(path[1:-1]).intersection(ex_nei)) for path in paths]
                    if min(len_paths) == 0:
                        pass
                    else:
                        path = paths[np.argmin(len_paths)]
                        for n in range(1, len(path) - 1):
                            A = torch.cat((A, torch.tensor(
                                [[path[n - 1]], [path[n]]])), dim=-1)
                            X[path[n], :] = X[node_index, :]  # av_feat_values
                            # TODO: eval this against av.values.

            # Apply model on new (X,A)
            if self.gpu:
                with torch.no_grad():
                    proba = self.model_to_explain(X.cuda(), A.cuda()).exp()[node_index]
            else:
                with torch.no_grad():
                    proba = self.model_to_explain(X, A).exp()[node_index]

            # Store predicted class label in fz
            if multiclass:
                fz[key] = proba
            else:
                fz[key] = proba[true_pred]

        return fz

    def graph_classification(self, graph_index, num_samples, D, z_, args_K, args_feat, true_pred):
        """ Construct z' from z and compute prediction f(z') for each sample z
            In fact, we build the dataset (z, f(z')), required to train the weighted linear model.
            Graph Classification task
        Args:
            Variables are defined exactly as defined in explainer function
            Note that adjacency matrices are dense (square) matrices (unlike node classification)
        Returns:
            (tensor): f(z') - probability of belonging to each target classes, for all samples z'
            Dimension (N * C) where N is num_samples and C num_classses.
        """
        # Store discarded nodes (z_j=0) for each sample z
        excluded_nei = {}
        for i in range(num_samples):
            # Excluded nodes' indexes
            nodes_id = []
            for j in range(len(self.neighbours)):
                if z_[i, self.F + j] == 0:
                    nodes_id+= self.neighbours[j]
            excluded_nei[i] = nodes_id
            # Dico with key = num_sample id, value = excluded neighbour index

        # Init
        fz = torch.zeros(num_samples)
        adj = deepcopy(self.graphs[graph_index])
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.features[graph_index].shape[1])
        else:
            feat = torch.vstack(self.features)
            av_feat_values = feat.mean(dim=0).mean(dim=0)
            # av_feat_values = np.mean(self.features[graph_index],axis=0)

        # Create new matrix A and X - for each sample ≈ reform z' from z
        #for (key, ex_nei) in tqdm(excluded_nei.items()):

        for (key, ex_nei) in (excluded_nei.items()):

            # Change adj matrix
            A = deepcopy(adj)
            A = to_dense_adj(A)[0]
            A[ex_nei, :] = 0
            A[:, ex_nei] = 0
            A = dense_to_sparse(A)[0]

            # Also change features of excluded nodes (optional)
            X = deepcopy(self.features[graph_index])
            for nei in ex_nei:
                X[nei] = av_feat_values

            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    proba = self.model_to_explain(X.cuda(), A.cuda())
            else:
                with torch.no_grad():
                    proba = self.model_to_explain(X, A)

            # Compute prediction
            fz[key] = proba[0][true_pred.item()]

        return fz

    def basic_default(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat,
                      discarded_feat_idx, multiclass, true_pred):
        """ Construct z' from z and compute prediction f(z') for each sample z
            In fact, we build the dataset (z, f(z')), required to train the weighted linear model.
            Does not deal with isolated 2 hops neighbours (or more)
        Args:
                Variables are defined exactly as defined in explainer function
        Returns:
                (tensor): f(z') - probability of belonging to each target classes, for all samples z'
                Dimension (N * C) where N is num_samples and C num_classses.
        """
        # Define an "average" feature vector - for discarded features
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.num_features)
        else:
            av_feat_values = self.features.mean(dim=0)

        # Define excluded_feat and excluded_nei for each z
        excluded_feat = {}
        excluded_nei = {}
        for i in range(num_samples):

            # Excluded features' indexes
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Excluded neighbors' indexes
            nodes_id = []
            for j in range(D):
                if z_[i, self.F + j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            excluded_nei[i] = nodes_id
            # Dico with key = num_sample id, value = excluded neighbour index

        # Init label f(z') for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        # Construct new matrices A and X for each sample - reform z from z'
        for (key, ex_nei), (_, ex_feat) in tqdm(zip(excluded_nei.items(), excluded_feat.items())):

            # For each excluded neighbour, retrieve the column index of its occurences
            # in the adj matrix - store them in positions (list)
            positions = []
            for val in ex_nei:
                pos = (self.graphs == val).nonzero()[:, 1].tolist()
                positions += pos
            positions = list(set(positions))
            A = np.array(self.graphs)
            A = np.delete(A, positions, axis=1)
            A = torch.tensor(A)

            # Change feature vector for node of interest
            X = deepcopy(self.features)
            X[node_index, ex_feat] = av_feat_values[ex_feat]

            # Discarded features approx
            # if args_feat != 'Null' and discarded_feat_idx != [] and len(self.neighbours) - len(ex_nei) < args_K:
            #     X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]

            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    proba = self.model_to_explain(X.cuda(), A.cuda()).exp()[node_index]
            else:
                with torch.no_grad():
                    proba = self.model_to_explain(X, A).exp()[node_index]

            # Store predicted class label in fz
            if multiclass:
                fz[key] = proba
            else:
                fz[key] = proba[true_pred]

            # Store predicted class label in fz
            if multiclass:
                fz[key] = proba
            else:
                fz[key] = proba[true_pred]

        return fz

    def neutral(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat,
                discarded_feat_idx, multiclass, true_pred):
        """ Construct z' from z and compute prediction f(z') for each sample z
            In fact, we build the dataset (z, f(z')), required to train the weighted linear model.
            Do not isolate nodes but set their feature vector to expected values
            Consider node features for node itself
        Args:
                Variables are defined exactly as defined in explainer function
        Returns:
                (tensor): f(z') - probability of belonging to each target classes, for all samples z
                Dimension (N * C) where N is num_samples and C num_classses.
        """
        # Initialise new node feature vectors and neighbours to disregard
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.num_features)
        else:
            av_feat_values = self.features.mean(dim=0)
        # or random feature vector made of random value across each col of X

        excluded_feat = {}
        excluded_nei = {}

        # Define excluded_feat and excluded_nei for each z'
        for i in tqdm(range(num_samples)):

            # Define new node features dataset (we only modify x_v for now)
            # Store index of features that are not sampled (z_j=0)
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Define new neighbourhood
            # Store index of neighbours that need to be isolated (not sampled, z_j=0)
            nodes_id = []
            for j in range(D):
                if z_[i, self.F + j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            # Dico with key = num_sample id, value = excluded neighbour index
            excluded_nei[i] = nodes_id

        # Init label f(z') for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        # Create new matrix A and X - for each sample ≈ reform z' from z
        for (key, ex_nei), (_, ex_feat) in zip(excluded_nei.items(), excluded_feat.items()):

            # Change feature vector for node of interest
            X = deepcopy(self.features)

            # For each excluded node, retrieve the column index of its occurences
            # in the adj matrix - store them in positions (list)
            A = self.graphs
            X[ex_nei, :] = av_feat_values.repeat(len(ex_nei), 1)
            # Set all excluded features to expected value for node index only
            X[node_index, ex_feat] = av_feat_values[ex_feat]
            if args_feat != 'Null' and discarded_feat_idx != [] and len(self.neighbours) - len(ex_nei) < args_K:
                X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]

            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    proba = self.model_to_explain(X.cuda(), self.graphs.cuda()).exp()[
                        node_index]
            else:
                with torch.no_grad():
                    proba = self.model_to_explain(X, self.graphs).exp()[
                        node_index]

            # Store predicted class label in fz
            if multiclass:
                fz[key] = proba
            else:
                fz[key] = proba[true_pred]

        return fz

        ################################

    ################################
    # Explanation Generator
    ################################

    def WLS(self, z_, weights, fz, multiclass, info):
        """ Weighted Least Squares Method
            Estimates shapley values via explanation model
        Args:
            z_ (tensor): binary vector representing the new instance
            weights ([type]): shapley kernel weights for z
            fz ([type]): prediction f(z') where z' is a new instance - formed from z and x
        Returns:
            [tensor]: estimated coefficients of our weighted linear regression - on (z, f(z'))
            Dimension (M * num_classes)
        """
        # Add constant term
        z_ = torch.cat([z_, torch.ones(z_.shape[0], 1)], dim=1)

        # WLS to estimate parameters
        try:
            tmp = np.linalg.inv(np.dot(np.dot(z_.T, np.diag(weights)), z_))
        except np.linalg.LinAlgError:  # matrix not invertible
            if info:
                print('WLS: Matrix not invertible')
            tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_)
            tmp = np.linalg.inv(
                tmp + np.diag(10 ** (-5) * np.random.randn(tmp.shape[1])))

        phi = np.dot(tmp, np.dot(
            np.dot(z_.T, np.diag(weights)), fz.detach().numpy()))

        # Test accuracy
        y_pred = z_.detach().numpy() @ phi
        if info:
            print('r2: ', r2_score(fz, y_pred))
            print('weighted r2: ', r2_score(fz, y_pred, sample_weight=weights))

        return phi[:-1], phi[-1]

    def WLR_sklearn(self, z_, weights, fz, multiclass, info):
        """Train a weighted linear regression
        Args:
            z_ (torch.tensor): dataset
            weights (torch.tensor): weights of each sample
            fz (torch.tensor): predictions for z_
        Return:
            tensor: parameters of explanation model g
        """
        # Convert to numpy
        weights = weights.detach().numpy()
        z_ = z_.detach().numpy()
        fz = fz.detach().numpy()

        # Fit weighted linear regression
        reg = LinearRegression()
        reg.fit(z_, fz, weights)
        y_pred = reg.predict(z_)

        # Assess perf
        if info:
            print('weighted r2: ', reg.score(z_, fz, sample_weight=weights))
            print('r2: ', r2_score(fz, y_pred))

        # Coefficients
        phi = reg.coef_
        base_value = reg.intercept_

        return phi, base_value

    def WLR_Lasso(self, z_, weights, fz, multiclass, info):
        """Train a weighted linear regression with lasso regularisation
        Args:
            z_ (torch.tensor): data
            weights (torch.tensor): weights of each sample
            fz (torch.tensor): y data

        Return:
            tensor: parameters of explanation model g

        """
        # Convert to numpy
        weights = weights.detach().numpy()
        z_ = z_.detach().numpy()
        fz = fz.detach().numpy()
        # Fit weighted linear regression
        reg = Lasso(alpha=0.01)
        # reg = Lasso()
        reg.fit(z_, fz, weights)
        y_pred = reg.predict(z_)
        # Assess perf
        if info:
            print('weighted r2: ', reg.score(z_, fz, sample_weight=weights))
            print('r2: ', r2_score(fz, y_pred))
        # Coefficients
        phi = reg.coef_
        base_value = reg.intercept_

        return phi, base_value

    def WLR(self, z_, weights, fz, multiclass, info):
        """Train a weighted linear regression
        Args:
            z_ (torch.tensor): data
            weights (torch.tensor): weights of each sample
            fz (torch.tensor): y data

        Return:
            tensor: parameters of explanation model g
        """
        # Define model
        if multiclass:
            our_model = LinearRegressionModel(
                z_.shape[1], self.data.num_classes)
        else:
            our_model = LinearRegressionModel(z_.shape[1], 1)
        our_model.train()

        # Define optimizer and loss function
        def weighted_mse_loss(input, target, weight):
            return (weight * (input - target) ** 2).mean()

        criterion = torch.nn.MSELoss()
        # optimizer = torch.optim.SGD(our_model.parameters(), lr=0.2)
        optimizer = torch.optim.Adam(our_model.parameters(), lr=0.001)

        # Dataloader
        train = torch.utils.data.TensorDataset(z_, fz, weights)
        train_loader = torch.utils.data.DataLoader(train, batch_size=1)

        # Repeat for several epochs
        for epoch in range(100):

            av_loss = []
            # for x,y,w in zip(z_,fz, weights):
            for batch_idx, (dat, target, w) in enumerate(train_loader):
                x, y, w = Variable(dat), Variable(target), Variable(w)

                # Forward pass: Compute predicted y by passing x to the model
                pred_y = our_model(x)

                # Compute loss
                loss = weighted_mse_loss(pred_y, y, w)
                # loss = criterion(pred_y,y)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Store batch loss
                av_loss.append(loss.item())
            if epoch % 10 == 0 and info:
                print('av loss epoch: ', np.mean(av_loss))

        # Evaluate model
        our_model.eval()
        with torch.no_grad():
            pred = our_model(z_)
        if info:
            print('weighted r2 score: ', r2_score(
                pred, fz, multioutput='variance_weighted'))
            if multiclass:
                print(r2_score(pred, fz, multioutput='raw_values'))
            print('r2 score: ', r2_score(pred, fz, weights))

        phi, base_value = [param.T for _,
                                       param in our_model.named_parameters()]
        phi = np.squeeze(phi, axis=1)
        return phi.detach().numpy().astype('float64'), base_value.detach().numpy().astype('float64')

    ################################
    # INFO ON EXPLANATIONS
    ################################

    def print_info(self, D, node_index, phi, feat_idx, true_pred, true_conf, base_value, multiclass):
        """
        Displays some information about explanations - for a better comprehension and audit
        """
        # Print some information
        print('Explanations include {} node features and {} neighbours for this node\
        for {} classes'.format(self.F, D, self.data.num_classes))

        # Compare with true prediction of the model - see what class should truly be explained
        print('Model prediction is class {} with confidence {}, while true label is {}'
              .format(true_pred, true_conf, self.data.y[node_index]))

        # Print base value
        print('Base value', base_value, 'for class ', true_pred.item())

        # Isolate explanations for predicted class - explain model choices
        if multiclass:
            pred_explanation = phi[true_pred, :]
        else:
            pred_explanation = phi

        # print('Explanation for the class predicted by the model:', pred_explanation)

        # Look at repartition of weights among neighbours and node features
        # Motivation for regularisation
        print('Weights for node features: ', sum(pred_explanation[:self.F]),
              'and neighbours: ', sum(pred_explanation[self.F:]))
        # print('Total Weights (abs val) for node features: ', sum(np.abs(pred_explanation[:self.F])),
        #      'and neighbours: ', sum(np.abs(pred_explanation[self.F:])))

        # Proportional importance granted to graph structure vs node features of v
        # print('Feature importance wrt explainable part: {} %'.format( 100 * sum(pred_explanation[:self.F]) / (true_pred.item())))
        # print('Node importance wrt explainable part: {} %'.format(100* sum(pred_explanation[self.F:]) / (true_pred.item())) )

        # Note we focus on explanation for class predicted by the model here, so there is a bias towards
        # positive weights in our explanations (proba is close to 1 everytime).
        # Alternative is to view a class at random or the second best class

        # Select most influential neighbours and/or features (+ or -)
        if self.F + D > 10:
            _, idxs = torch.topk(torch.from_numpy(np.abs(pred_explanation)), 6)
            vals = [pred_explanation[idx] for idx in idxs]
            influential_feat = {}
            influential_nei = {}
            for idx, val in zip(idxs, vals):
                if idx.item() < self.F:
                    influential_feat[feat_idx[idx]] = val
                else:
                    influential_nei[self.neighbours[idx - self.F]] = val
            print('Most influential features: ',
                  len([(item[0].item(), item[1].item()) for item in list(influential_feat.items())]),
                  'and neighbours', len([(item[0].item(), item[1].item()) for item in list(influential_nei.items())]))

        # Most influential features splitted bewteen neighbours and features
        if self.F > 5:
            _, idxs = torch.topk(torch.from_numpy(
                np.abs(pred_explanation[:self.F])), 3)
            vals = [pred_explanation[idx] for idx in idxs]
            influential_feat = {}
            for idx, val in zip(idxs, vals):
                influential_feat[feat_idx[idx]] = val
            print('Most influential features: ', [
                (item[0].item(), item[1].item()) for item in list(influential_feat.items())])

        # Most influential features splitted bewteen neighbours and features
        if D > 5 and self.M != self.F:
            _, idxs = torch.topk(torch.from_numpy(
                np.abs(pred_explanation[self.F:])), 3)
            vals = [pred_explanation[self.F + idx] for idx in idxs]
            influential_nei = {}
            for idx, val in zip(idxs, vals):
                influential_nei[self.neighbours[idx]] = val
            print('Most influential neighbours: ', [
                (item[0].item(), item[1].item()) for item in list(influential_nei.items())])

    def vizu(self, edge_mask, node_index, phi, predicted_class, hops, multiclass):
        """ Vizu of important nodes in subgraph around node_index
        Args:
            edge_mask ([type]): vector of size data.edge_index with False
                                            if edge is not included in subgraph around node_index
            node_index ([type]): node of interest index
            phi ([type]): explanations for node of interest
            predicted_class ([type]): class predicted by model for node of interest
            hops ([type]):  number of hops considered for subgraph around node of interest
            multiclass: if we look at explanations for all classes or only for the predicted one
        """
        if multiclass:
            phi = torch.tensor(phi[predicted_class, :])
        else:
            phi = torch.from_numpy(phi).float()

        # Replace False by 0, True by 1 in edge_mask
        mask = edge_mask.int().float()

        # Identify one-hop subgraph around node_index
        one_hop_nei, _, _, _ = torch_geometric.utils.k_hop_subgraph(
            node_index, 1, self.graphs, relabel_nodes=True,
            num_nodes=None)
        # true_one_hop_nei = one_hop_nei[one_hop_nei != node_index]

        # Attribute phi to edges in subgraph bsed on the incident node phi value
        for i, nei in enumerate(self.neighbours):
            list_indexes = (self.graphs[0, :] == nei).nonzero()
            for idx in list_indexes:
                # Remove importance of 1-hop neighbours to 2-hop nei.
                if nei in one_hop_nei:
                    if self.graphs[1, idx] in one_hop_nei:
                        mask[idx] = phi[self.F + i]
                elif mask[idx] == 1:
                    mask[idx] = phi[self.F + i]
            # mask[mask.nonzero()[i].item()]=phi[i, predicted_class]

        # Set to 0 importance of edges related to node_index
        mask[mask == 1] = 0

        # Increase coef for visibility and consider absolute contribution
        mask = torch.abs(mask)
        mask = mask / sum(mask)

        # Vizu nodes
        ax, G = visualize_subgraph(self.model_to_explain,
                                   node_index,
                                   self.graphs,
                                   mask,
                                   hops,
                                   y=self.data.y,
                                   threshold=None)

        plt.savefig('results/GS1_{}_{}_{}'.format(self.data.name,
                                                  self.model_to_explain.__class__.__name__,
                                                  node_index),
                    bbox_inches='tight')

        # Other visualisation
        G = denoise_graph(self.data, mask, phi[self.F:], self.neighbours,
                          node_index, feat=None, label=self.data.y, threshold_num=10)

        log_graph(G,
                  identify_self=True,
                  nodecolor="label",
                  epoch=0,
                  fig_size=(4, 3),
                  dpi=300,
                  label_node_feat=False,
                  edge_vmax=None,
                  args=None)

        plt.savefig('results/GS_{}_{}_{}'.format(self.data.name,
                                                 self.model_to_explain.__class__.__name__,
                                                 node_index),
                    bbox_inches='tight')


class SHAP():
    """ KernelSHAP explainer - adapted to GNNs
    Explains only node features
    """

    def __init__(self, data, model, gpu=False):
        self.model_to_explain = model
        self.data = data
        self.gpu = gpu
        # number of nonzero features - for each node index
        self.M = self.data.num_features
        self.neighbours = None
        self.F = self.M

        #self.model_to_explain.eval()

    def explain(self, node_index=0, hops=2, num_samples=10, info=True, multiclass=False, *unused):
        """
        :param node_index: index of the node of interest
        :param hops: number k of k-hop neighbours to consider in the subgraph around node_index
        :param num_samples: number of samples we want to form GraphSVX's new dataset
        :return: shapley values for features that influence node v's pred
        """
        # Compute true prediction of model, for original instance
        with torch.no_grad():
            if self.gpu:
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
                self.model_to_explain = self.model_to_explain.to(device)
                true_conf, true_pred = self.model_to_explain(
                    x=self.features.cuda(), edge_index=self.graphs.cuda()).exp()[node_index][0].max(dim=0)
            else:
                true_conf, true_pred = self.model_to_explain(
                    x=self.features, edge_index=self.graphs).exp()[node_index][0].max(dim=0)

        # Determine z => features whose importance is investigated
        # Decrease number of samples because nodes are not considered
        num_samples = num_samples // 3

        # Consider all features (+ use expectation like below)
        # feat_idx = torch.unsqueeze(torch.arange(self.F), 1)

        # Sample z - binary vector of dimension (num_samples, M)
        z_ = torch.empty(num_samples, self.M).random_(2)
        # Compute |z| for each sample z
        s = (z_ != 0).sum(dim=1)

        # Define weights associated with each sample using shapley kernel formula
        weights = self.shapley_kernel(s)

        # Create dataset (z, f(z')), stored as (z_, fz)
        # Retrive z' from z and x_v, then compute f(z')
        fz = self.compute_pred(node_index, num_samples,
                               z_, multiclass, true_pred)

        # OLS estimator for weighted linear regression
        phi, base_value = self.OLS(z_, weights, fz)  # dim (M*num_classes)

        return  phi

    def shapley_kernel(self, s):
        """
        :param s: dimension of z' (number of features + neighbours included)
        :return: [scalar] value of shapley value
        """
        shap_kernel = []
        # Loop around elements of s in order to specify a special case
        # Otherwise could have procedeed with tensor s direclty
        for i in range(s.shape[0]):
            a = s[i].item()
            # Put an emphasis on samples where all or none features are included
            if a == 0 or a == self.M:
                shap_kernel.append(1000)
            elif scipy.special.binom(self.M, a) == float('+inf'):
                shap_kernel.append(1 / self.M)
            else:
                shap_kernel.append(
                    (self.M - 1) / (scipy.special.binom(self.M, a) * a * (self.M - a)))
        return torch.tensor(shap_kernel)

    def compute_pred(self, node_index, num_samples, z_, multiclass, true_pred):
        """
        Variables are exactly as defined in explainer function, where compute_pred is used
        This function aims to construct z' (from z and x_v) and then to compute f(z'),
        meaning the prediction of the new instances with our original model.
        In fact, it builds the dataset (z, f(z')), required to train the weighted linear model.
        :return fz: probability of belonging to each target classes, for all samples z'
        fz is of dimension N*C where N is num_samples and C num_classses.
        """
        # This implies retrieving z from z' - wrt sampled neighbours and node features
        # We start this process here by storing new node features for v and neigbours to
        # isolate
        X_v = torch.zeros([num_samples, self.F])

        # Init label f(z') for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        # Do it for each sample
        for i in range(num_samples):

            # Define new node features dataset (we only modify x_v for now)
            # Features where z_j == 1 are kept, others are set to 0
            for j in range(self.F):
                if z_[i, j].item() == 1:
                    X_v[i, j] = 1

            # Change feature vector for node of interest
            X = deepcopy(self.features)
            X[node_index, :] = X_v[i, :]

            # Apply model on (X,A) as input.
            with torch.no_grad():
                if self.gpu:
                    proba = self.model_to_explain(x=X.cuda(), edge_index=self.graphs.cuda()).exp()[
                        node_index]
                else:
                    proba = self.model_to_explain(x=X, edge_index=self.graphs).exp()[
                        node_index]
            # Multiclass
            if not multiclass:
                fz[i] = proba[true_pred]
            else:
                fz[i] = proba

        return fz

    def OLS(self, z_, weights, fz):
        """
        :param z_: z - binary vector
        :param weights: shapley kernel weights for z
        :param fz: f(z') where z is a new instance - formed from z and x
        :return: estimated coefficients of our weighted linear regression - on (z, f(z'))
        phi is of dimension (M * num_classes)
        """
        # Add constant term
        z_ = torch.cat([z_, torch.ones(z_.shape[0], 1)], dim=1)

        # WLS to estimate parameters
        try:
            tmp = np.linalg.inv(np.dot(np.dot(z_.T, np.diag(weights)), z_))
        except np.linalg.LinAlgError:  # matrix not invertible
            tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_)
            tmp = np.linalg.inv(
                tmp + np.diag(0.00001 * np.random.randn(tmp.shape[1])))
        phi = np.dot(tmp, np.dot(
            np.dot(z_.T, np.diag(weights)), fz.cpu().detach().numpy()))

        # Test accuracy
        # y_pred=z_.detach().numpy() @ phi
        #	print('r2: ', r2_score(fz, y_pred))
        #	print('weighted r2: ', r2_score(fz, y_pred, weights))

        return phi[:-1], phi[-1]
