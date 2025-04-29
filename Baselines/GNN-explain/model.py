import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch_geometric
from torch_geometric.nn import SAGEConv, global_max_pool, GraphConv, TopKPooling, GCNConv, GatedGraphConv, GATConv,  TAGConv,  SGConv, global_mean_pool
from typing import List, Tuple, Callable

from torch_geometric.data import Data, DataLoader

class GCNet(nn.Module):
    """
    Graph convolution network, take multi-graph described as an adjacency 3 dimensional tensor.
    """

    def __init__(self, input_dim:int, hidden_dims: List[int], decoder_dims:List[int], output_dim:int, loss=None, **_):
        super(GCNet, self).__init__()

        ratio=[1 for _ in range(len(hidden_dims))]
        self.loss_fn = loss
        dims = [input_dim] + hidden_dims
        #self.layers = nn.ModuleList([GCNLayer(dims[i], dims[i + 1]) for i in range(len(hidden_dims) - 1)])
        self.layers = nn.ModuleList([GraphConv(dims[i], dims[i + 1], bias=True) for i in range(len(hidden_dims))])
        #self.layers = nn.ModuleList([GATConv(dims[i], dims[i + 1], bias=False) for i in range(len(hidden_dims) - 1)])
        self.poolings = nn.ModuleList([TopKPooling(dims[i+1], ratio=ratio[i]) for i in range(len(hidden_dims))])


        #self.lin1 = nn.Linear(hidden_dims[-1], hidden_dims[-1], bias=False)
        #self.lin2 = nn.Linear(hidden_dims[-1], output_dim, bias=False)

        decdims = [hidden_dims[-1]] + decoder_dims

        self.decoder = nn.ModuleList([nn.Linear(decdims[i], decdims[i + 1], bias=False) for i in range(len(decdims)-1)])
        self.lin = nn.Linear(decdims[-1], output_dim, bias=False)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data:Data):# x, adj):
        return self.internal_forward(data)[-1][0]

    def internal_forward(self, data:Data, register=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        output=list()
        for i, (lay,pool) in enumerate(zip(self.layers, self.poolings)):
            x = lay(x, edge_index)
            if register:
                x.register_hook(register)
            """if self.training:
                x=self.dropout(x)
            else:
                x = x * 2"""
            x = fun.leaky_relu(x, negative_slope=0.01)
            #x = fun.dropout(x,0.5, training=self.training)

            #x, edge_index,_, batch,_,_ = pool(x, edge_index, None, batch)
            if i>2 and (i - 1) % 2 == 0:
                x+=output[-2][0]
            output.append((x,batch))

        # x = x.max(dim=1)[0]

        #x = global_mean_pool(x, batch)
        x = global_mean_pool(x, batch)

        for lay in self.decoder:
            x = fun.dropout(x,0.5, training=self.training)
            x = lay(x)
            output.append((x,None))
            x = fun.leaky_relu(x, negative_slope=0.001)
        x = self.lin(x)
        x=x.squeeze(0)
        x=fun.softmax(x, -1)
        #x = torch.sigmoid(x).squeeze(0)
        output.append((x,None))
        return output  # fun.softmax(x, -1)

    def loss(self, pred:torch.Tensor, target:torch.Tensor):
        if self.loss_fn is not None:
            return self.loss_fn(pred, target)
        return torch.nn.MSELoss()(pred, target)


class SDNE(nn.Module):
    def __init__(self,input_dim, encoder_dims:List[int], embedding_dim:int, decoder_dims:List[int],loss=None):
        super(SDNE, self).__init__()

        #ratio=[1 for _ in range(len(hidden_dims))]
        self.loss_fn = loss
        encoder_dims = [input_dim] + encoder_dims + [embedding_dim]
        decoder_dims =  [embedding_dim] + decoder_dims + [input_dim]
        self.encoder_layers = nn.ModuleList([GraphConv(encoder_dims[i], encoder_dims[i + 1], bias=True) for i in range(len(encoder_dims)-1)])
        self.decoder_layers = nn.ModuleList([GraphConv(decoder_dims[i], decoder_dims[i + 1], bias=True) for i in range(len(decoder_dims)-1)])


    def _forward(self, data:Data, register=False):
        output = self._encoder(data,register)
        embedding =output[-1][0]
        output = self._decoder(data, output,register)
        return output, embedding

    def forward(self, data:Data, register=False):
        out, emb= self._forward(data,register=register)
        return out[-1][0], emb


    def _encoder(self, data:Data, register=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        output=list()
        for i, lay in enumerate(self.encoder_layers):
            x = lay(x, edge_index)
            if register:
                x.register_hook(register)
            x = fun.leaky_relu(x, negative_slope=0.01)
            output.append((x,batch))

        return output

    def encoder(self, data:Data, register=False):
        return self._encoder(data,register)[-1][0]

    def _decoder(self, data, output=list(), register=False):
        x = output[-1][0]
        edge_index, batch = data.edge_index, data.batch
        for i, lay in enumerate(self.decoder_layers):
            x = lay(x, edge_index)
            if register:
                x.register_hook(register)
            x = fun.leaky_relu(x, negative_slope=0.01)
            output.append((x,batch))
        return output

    def decoder(self, x, data, register=False):
        return self._decoder(data, [[x]], register=register)[-1][0]

    def loss(self, pred:torch.Tensor, Y):
        #if self.loss_fn is not None:
        #    return self.loss_fn(pred[0], target)
        l1st = self.loss_1st(pred, Y)
        lmse = self.mseloss(pred, Y)
        return l1st+lmse

    @staticmethod
    def mseloss(pred, target):
        return torch.nn.MSELoss()(pred[0], target.x)
    @staticmethod
    def loss_1st( H, target):
        L1 = torch_geometric.utils.get_laplacian(target.edge_index,num_nodes=H[1].size()[0])
        L = torch_geometric.utils.to_dense_adj(L1[0], edge_attr=L1[1])[0]
        return 2* torch.trace(torch.chain_matmul(H[1].t(),L,H[1]))
