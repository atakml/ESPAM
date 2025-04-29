from graphxai.gnn_models.graph_classification.gcn import GCN_3layer
from graphxai.gnn_models.graph_classification.utils import train, test
from torch.optim import Adam
from ExplanationEvaluation.models.model_selector import model_selector
from ExplanationEvaluation.configs.selector import Selector
from torch.nn import CrossEntropyLoss
import torch
from os import listdir
from os.path import isfile, join

import ExplanationEvaluation.explainers.SVXExplainer as svxm
from importlib import reload
from tqdm import tqdm
reload(svxm)
from graphxai.utils import Explanation
from GModel import GModel
from torch_geometric.data import Data
import random
import numpy as np
import pickle 
import argparse
from dgl.nn import SubgraphX
from torch_geometric.utils import to_dgl, from_dgl
from dgl import DGLGraph


seed = 4
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU
np.random.seed(seed)
random.seed(seed)

explain_dict = {}
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help="Name of the dataset")

parser.add_argument("--i", type=int, nargs='+', required=True)
args = parser.parse_args()
index = args.i[0]


_explainer = "actexplainer"

_folder = 'replication' # One of: replication, extension

dataset_name = args.dataset
explain_dict[dataset_name] = {}
config_path = f"ExplanationEvaluation/configs/{_folder}/explainers/{_explainer}/{dataset_name}.json"
radius = 3
config = Selector(config_path)
config = config.args.explainer
model, checkpoint = model_selector(config.model,
                                    config.dataset,
                                    pretrained=True,
                                    return_checkpoint=True)
model.eval()





class GStarModel(torch.nn.Module):
    def __init__(self, forward_function):
        super(GStarModel, self).__init__()
        self.forward_function = forward_function
    def forward(self, data, feat):
        if isinstance(data, DGLGraph):
            data = from_dgl(data)
        res = self.forward_function(feat, data.edge_index)
        return res.softmax(dim=-1)

gstar_model = GStarModel(model)

subgraph_explainer = SubgraphX(gstar_model, 3, shapley_steps=10)


with open(f"{dataset_name}/subgraphindex/{index}.pkl", "rb") as file:
    data = pickle.load(file)
target_class = torch.argmax(gstar_model(data, data.x)[0])
subgraph_explanation = subgraph_explainer.explain_graph(to_dgl(data), data.x, target_class)
with open(f"{dataset_name}/subgraph/{index}.pkl", "wb") as file:
    pickle.dump(subgraph_explanation, file)
