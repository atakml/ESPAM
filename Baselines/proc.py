import torch
import pickle 
from ExplanationEvaluation.models.model_selector import model_selector
from ExplanationEvaluation.configs.selector import Selector
import argparse
import random
import numpy as np
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import warnings
warnings.filterwarnings('ignore')
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help="Name of the dataset")

parser.add_argument("--i", type=int, nargs='+', required=True)
args = parser.parse_args()
index = args.i[0]
dataset_name = args.dataset

from tqdm import tqdm
explain_dict = {}
_explainer = "actexplainer"

_folder = 'replication' # One of: replication, extension

explain_dict[dataset_name] = {}
config_path = f"ExplanationEvaluation/configs/{_folder}/explainers/{_explainer}/{dataset_name}.json"
radius = 3
config = Selector(config_path)
config = config.args.explainer
print(dataset_name)
model, checkpoint = model_selector(config.model,
                                    config.dataset,
                                    pretrained=True,
                                    return_checkpoint=True)
model.eval()
seed = 4
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU
np.random.seed(seed)
random.seed(seed)



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#from graphxai.utils import Explanation
from torch_geometric.data import Data
def model_wrapper(model):
    model = model.to("cuda:0")
    def model_score(x, edge_index, edge_weights=None):
        if edge_weights is None:
            return model(x.to("cuda:0"), edge_index.to("cuda:0"), None)
        new_edge_index = edge_index[:, edge_weights.to("cuda:0")]
        return model(x.to("cuda:0"), new_edge_index.to("cuda:0"), None)
    return model_score
model_hfid = model_wrapper(model)


from GStarX.gstarx import GStarX
class GStarModel(torch.nn.Module):
    def __init__(self, forward_function):
        super(GStarModel, self).__init__()
        self.forward = forward_function
gstar_model = model#GStarModel(model_hfid)




with open(f"{dataset_name}/gstarindex/{index}.pkl", "rb") as file:
    data = pickle.load(file)
gstar_explainer = GStarX(gstar_model, device="cuda:0", payoff_type="prob")
gstar_explanation = gstar_explainer.explain(data)
with open(f"{dataset_name}/gstar/{index}.pkl", "wb") as file:
    pickle.dump(gstar_explanation, file)

print(f"index {index} done")
