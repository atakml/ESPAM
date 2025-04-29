import numpy as np
import os
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import networkx as nx
import code

import configs
from GNN_Transformer.optimal_transport_for_gnn.src.OTBFS import evaluate_score_from_file
from GNN_Transformer.optimal_transport_for_gnn.src.parse_json import get_ego_networks, add_features_from_matrix, add_edges
from utils import convert_conjunction_to_dict, get_support_indices, convert_subgroup_to_dict, build_dataset_from_file, \
    load_mol_dataset

def draw_histogram(rule, layer,activation_matrix, graph_indices, bert_func):
    ego_path = f"ego_networks/{rule}_{layer}"
    files = os.listdir(ego_path)
    ego_scores = []
    for fl in files:
        ego_scores.append(evaluate_score_from_file(configs.config_file, ego_path + "/" + fl, 1, rule, layer, activation_matrix, "cosine", bert_func, configs.prediction_file, graph_indices))
    plt.clf()
    plt.hist(ego_scores, bins=10)
    plt.savefig(f"hist_{rule}_{layer}")


def generate_ego_graphs(rule, layer, activation_matrix, graph_indices):
    
    with open(f"GNN_Transformer/rule_supports/support{str(convert_subgroup_to_dict(rule[1].Rule))}_{layer}.json") as js:
        data = json.load(js)
    debug_rule = {3: 1, 58: 1, 6: 1, 8: 1} 
    os.system(f"mkdir \"ego_networks/{str(convert_subgroup_to_dict(rule[1].Rule))}_{layer}\"")
    ego_scores = []
    for key, val in data.items():
        for i in range(len(val)):
            graph_list = val[i][1]
            new_nx = nx.Graph()
            new_nx.add_node(0)
            edge_index = graph_list[1]
            edge_types = graph_list[2]
            add_edges(new_nx, edge_index, edge_types)
            features_matrix = np.array(graph_list[0])
            add_features_from_matrix(new_nx, features_matrix)
            ego_networks = get_ego_networks(new_nx, activation_matrix.to_numpy(), val[i][0], (layer+1)*6, rule, graph_indices)
            for j, ego_graph in enumerate(ego_networks):
                pickle_file = f"ego_networks/{str(convert_subgroup_to_dict(rule[1].Rule))}_{layer}/{i}_{j}.pkl"
                with open(pickle_file, "wb") as pkl:
                    pickle.dump((ego_graph, j), pkl)


dataset, bert_func = load_mol_dataset(configs.prediction_file, configs.seq_file, configs.bert_file)
for layer in range(5):
    rule_file = f"GNN_Transformer/rules/rules_si_divided_10_{layer}.csv"
    rules = pd.read_csv(rule_file)
    file_to_read_data = configs.file_to_read_data
    file_to_read_data = file_to_read_data[:-5] + str(layer) + ".pkl"
    with open(file_to_read_data, "rb") as pkl:
        activation_matrix, graph_indices, labels = pickle.load(pkl)
    for rule in rules.iterrows():
        draw_histogram(convert_subgroup_to_dict(rule[1].Rule), layer, activation_matrix, graph_indices, bert_func)
