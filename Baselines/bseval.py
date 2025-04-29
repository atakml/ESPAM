import pickle 
import torch
from torch_geometric.data import Data
from ExplanationEvaluation.models.model_selector import model_selector
from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.tasks.replication import get_classification_task, to_torch_graph
from importlib import reload
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
from torch_geometric.utils import to_networkx
import networkx as nx
import gc


def fidelity(data, selected_nodes, model, target_class, remove_mask_function):
    #z = torch.ones(data.x.shape[0]).bool()
    #z[center_indices] = False
    x, edge_index = data.x, data.edge_index
    complement_mask = ~selected_nodes
    masked_graph = remove_mask_function(data, complement_mask)#features_to_graph(data, z, 1)
    return model(x, edge_index).softmax(-1)[0][target_class] - model(masked_graph.x, masked_graph.edge_index).softmax(-1)[0][target_class]


def infidelity(data, selected_nodes, model, target_class, remove_mask_function):
    #z = torch.zeros(data.x.shape[0]).bool()
    #z[center_indices] = True
    x, edge_index = data.x, data.edge_index

    complement_mask = selected_nodes
    masked_graph = remove_mask_function(data, complement_mask)#features_to_graph(data, z, 1)
    return model(x, edge_index).softmax(-1)[0][target_class] - model(masked_graph.x, masked_graph.edge_index).softmax(-1)[0][target_class]


def sparsity(data, selected_nodes, remove_mask_function):
    #data = from_smiles_to_data(smiles)
    #z = torch.zeros(data.x.shape[0]).bool()
    #z[center_indices] = True
    #complement_mask = ~selected_nodes
    #print(complement_mask)

    masked_graph = remove_mask_function(data, selected_nodes)#features_to_graph(data, z, 1)
    return 1 - (masked_graph.true_edges)/(data.edge_index.shape[1])





def creat_mask_mol_data(data, mask, is_edge=False, inv=True): 
    mask = mask.to(bool)
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    if not is_edge:
        if inv:
            x[~mask] = torch.zeros_like(x[~mask])
        else:
            x = x[mask]
        edge_mask = mask[edge_index[0]]&mask[edge_index[1]]
        if inv:
            edge_index = edge_index[:, edge_mask]

        else:
            edge_index = edge_index[:, edge_mask]
    else:
        edge_index = edge_index[:, mask]
        edge_mask = mask
        node_mask = torch.zeros(data.x.shape[0])
        node_mask[edge_index[0]] = 1
        node_mask[edge_index[1]] = 1
        x[~node_mask.bool()] = torch.zeros_like(x[~node_mask.bool()])
    res = Data(x=x, edge_index=edge_index, true_edges=sum(edge_mask), node_mask_size =node_mask.sum() if is_edge else mask.sum())
    return res



#####################
#####################
#####################

       

def mask_compute(data, values,  model, target_class, mask_creator_function, min_sparsity):
    number_of_nodes = data.x.shape[0]
    G = to_networkx(data)
    G.remove_edges_from(nx.selfloop_edges(G))
    isolated_nodes = list(nx.isolates(G))
    number_of_nodes -= len(isolated_nodes)
    best_hfid = -100
    fid_make_mask = lambda data, mask: creat_mask_mol_data(data, mask, inv=True)
    for i in range(1,number_of_nodes):
        mask = mask_creator_function(data, (values, number_of_nodes), i)
        fid = fidelity(data, mask, model, target_class, fid_make_mask)
        infid = infidelity(data, mask, model, target_class, fid_make_mask)
        spars = 1 -i/data.x.shape[0]
        if spars < min_sparsity:
            break
        n_fid = fid*spars
        n_infid = infid*(1 - spars)
        hfid = ((1 + n_fid)*(1 - n_infid))/(2+ n_fid - n_infid)
        best_hfid = max(hfid, best_hfid)
    return best_hfid



def top_k_percent_nodes(values, k):
    if len(values) == 2:
        values, num_nodes = values
    else:
        num_nodes = 0
    values = torch.tensor(values)
    number_of_nodes = values.shape[0]
    num_nodes = num_nodes if num_nodes else number_of_nodes
    _, indices = torch.sort(values[:num_nodes], descending=True)
    selected_indices = indices[:k]
    mask = torch.zeros(number_of_nodes)
    mask[selected_indices] = 1
    return mask.to(bool)

    
    



def gnn_explainer_mask(data):
    edge_mask = data.edge_mask
    edge_mask = edge_mask > 0.5
    return edge_mask 

def edge_soft_mask(values, number_of_edges):
    values = torch.tensor(values)
    number_of_nodes = values.shape[0]
    _, indices = torch.sort(values, descending=True)
    selected_indices = indices[:number_of_edges]
    mask = torch.zeros(values.shape[0])
    mask[selected_indices] = 1
    return mask.to(bool)

def subgraph_mask(number_of_nodes, indices):
    mask = torch.zeros(number_of_nodes)
    mask[indices] = 1
    mask = mask.bool()
    return mask 


def hfidelity(data, mask, model, target_class, mask_func):
    number_of_nodes = data.x.shape[0]
    number_of_mask_nodes = mask_func(data, mask).node_mask_size
    fid = fidelity(data, mask, model, target_class, mask_func)
    infid = infidelity(data, mask, model, target_class, mask_func)
    spars = 1 -number_of_mask_nodes/number_of_nodes
    n_fid = fid*spars
    n_infid = infid*(1 - spars)
    hfid = ((1 + n_fid)*(1 - n_infid))/(2+ n_fid - n_infid)
    return hfid




dataset_name = "ba2"

# Load pretrained models
_dataset = "ba2motifs"
_explainer = "actexplainer"

_folder = 'replication' # One of: replication, extension

config_path = f"ExplanationEvaluation/configs/{_folder}/explainers/{_explainer}/{_dataset}.json"
radius = 3
config = Selector(config_path)
config = config.args.explainer
model, checkpoint = model_selector(config.model,
                                    config.dataset,
                                    pretrained=True,
                                    return_checkpoint=True)
if config.eval_enabled:
    model.eval()
    
graphs, features, labels, train_mask, _, test_mask = load_dataset(config.dataset)
task = get_classification_task(graphs)

graphs = [g for i, g in enumerate(graphs) if train_mask[i]]
features = [f for i, f in enumerate(features) if train_mask[i]]
labels = [l for i, l in enumerate(labels) if train_mask[i]]


features = torch.tensor(features)
labels = torch.tensor(labels)
graphs = to_torch_graph(graphs, task)
print(features.shape, labels.shape, len(graphs))
dataset = [Data(x=features[i], edge_index=graphs[i], y=labels[i]) for i in range(len(graphs))]

    
    

exp_name = "subgraph"
#dataset_name = "aids"
res_dict = {}

with open(f"{_dataset}/{exp_name}.pkl", "rb") as file:
    explain_dict = pickle.load(file)
explainer = exp_name
hfid_dict = {}
min_sparsity = 0.5
hfid_val = {}
hfid_dict = {}
mask_func_edge = lambda x, y: creat_mask_mol_data(x, y, is_edge=True)
mask_func_node = lambda x, y: creat_mask_mol_data(x, y, is_edge=False)
explanation_data = explain_dict#exp1
for index, explanations in tqdm(explanation_data.items()):
    number_of_nodes = features[index].shape[0]
    target_class = model(features[index], graphs[index])[0].argmax().item()
    data = dataset[index].clone()
    if explainer not in hfid_dict:
        hfid_dict[explainer] = 0
        hfid_val[explainer] = []
    if explainer == "GNNExplainer":
        mask = gnn_explainer_mask(explanations)
        hfid = hfidelity(data, mask, model, target_class, mask_func_edge)
        hfid_val[explainer].append(hfid)
        hfid_dict[explainer] += hfid
    elif explainer == "PGExplainer":
        values = explanations[1]
        max_h_fid = 0 
        for i in range(values.shape[0]):
            mask = edge_soft_mask(values, i)
            hfid = hfidelity(data, mask, model, target_class, mask_func_edge)
            max_h_fid = max(hfid, max_h_fid)
            sparsity = 1 - mask_func_edge(data, mask).node_mask_size/data.x.shape[0]
            if sparsity < min_sparsity:
                break
        hfid_val[explainer].append(max_h_fid)  
        hfid_dict[explainer] += max_h_fid
                
    elif explainer == "PGM":
        mask = explanations[1].bool()
        hfid = hfidelity(data, mask, model, target_class, mask_func_edge)
        hfid_val[explainer].append(hfid)
        hfid_dict[explainer] += hfid
    elif explainer == "gstar":
        explanations = torch.tensor(explanations)
        max_h_fid = 0
        for i in range(number_of_nodes):
            mask = top_k_percent_nodes(explanations, i)
            hfid = hfidelity(data, mask, model, target_class, mask_func_node)
            max_h_fid = max(hfid, max_h_fid)
            sparsity =  1 - i/number_of_nodes
            if sparsity < min_sparsity:
                break
        hfid_val[explainer].append(max_h_fid)
        hfid_dict[explainer] += max_h_fid
    elif explainer == "svx":
        values = explanations[:number_of_nodes]
        if values.shape[0] < number_of_nodes:
            values = torch.cat([values, torch.zeros(number_of_nodes - values.shape[0])])
        for i in range(number_of_nodes):
            mask = top_k_percent_nodes(values, i)
            hfid = hfidelity(data, mask, model, target_class, mask_func_node)
            max_h_fid = max(hfid, max_h_fid)
            sparsity =  1 - i/number_of_nodes
            if sparsity < min_sparsity:
                break
        hfid_val[explainer].append(max_h_fid)
        hfid_dict[explainer] += max_h_fid

    elif explainer == "subgraph":
        mask = subgraph_mask(number_of_nodes, explanations)
        hfid = hfidelity(data, mask, model, target_class, mask_func_node)
        hfid_val[explainer].append(hfid)
        hfid_dict[explainer] += hfid
        
    elif explainer == "GradCAM":
        values = explanations.node_imp
        max_h_fid = 0
        for i in range(number_of_nodes):
            mask = top_k_percent_nodes(values, i)
            hfid = hfidelity(data, mask, model, target_class, mask_func_node)
            max_h_fid = max(hfid, max_h_fid)
            sparsity =  1 - i/number_of_nodes
            if sparsity < min_sparsity:
                break
        hfid_val[explainer].append(max_h_fid)
        hfid_dict[explainer] += max_h_fid
    else: 
            pass
for explainer in hfid_dict.keys():
    res_dict[explainer] = hfid_dict[explainer]/len(explanation_data)
    with open(f"{dataset_name}_res_{exp_name}_dict.pkl", "wb") as file:
        pickle.dump(res_dict, file)
# Clear variables and free up memory
del explain_dict, hfid_dict, hfid_val, explanation_data, data, mask, target_class
torch.cuda.empty_cache()
gc.collect()
