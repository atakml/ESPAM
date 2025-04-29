import os.path

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import code
import numpy as np

from GNN_Transformer.optimal_transport_for_gnn.src.OTBFS import parse_median_pickle_file
import configs
from utils import convert_subgroup_to_dict

atoms = {2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In'}

def save_figure(graph, name, center):
    plt.clf()
    labels = {node: atoms[int(list(graph.nodes[node]["type"])[0])] for i, node in enumerate(graph.nodes())}
    nodes = [node for node in graph.nodes()]
    colors = []
    for node in graph.nodes():
        if graph.nodes[node]['center']:
            color = "red"
        else:
            color = "blue"
        colors.append(color)
    nx.draw_kamada_kawai(graph, nodelist=nodes, labels=labels, with_labels=True, node_color=colors)
    plt.savefig(name)


def f(x):
    return {x[0]}


def change_graph_attribute(graph):
    label = nx.get_node_attributes(graph, "label")
    label = {key: f(value) for key, value in label.items()}
    nx.set_node_attributes(graph, label, "type")
    return graph


for layer in range(5):
    # center_median is the key of the node not index
    rule_file = f"GNN_Transformer/rules/rules_si_divided_10_{layer}.csv"
    rules = pd.read_csv(rule_file)
    for rl in rules.iterrows():
        rule = convert_subgroup_to_dict(rl[1].Rule)
        suffix = f"_{rule}_{layer}"
        graph_median_file = f"median_graphs/graph_median{suffix}.pkl"
        graph_median_final_file = f"representations/final_ot{suffix}.pkl"
        graph_random_file = f"random_graphs/graph_random{suffix}.pkl"
        final_random_file = f"representations/final_random{suffix}.pkl"
        graph_median, index_median, center_median = parse_median_pickle_file(graph_median_file)
        graph_median = change_graph_attribute(graph_median)
        if not os.path.exists(f"figures/{suffix[1:]}"):
            os.system(f"mkdir \"figures/{suffix[1:]}\"")
        save_figure(graph_median, f"figures/{suffix[1:]}/graph_median_{suffix[1:]}_median", center_median)
        try:
            with open(graph_median_final_file, "rb") as pkl:
                graph_median_final = change_graph_attribute(pickle.load(pkl)[0])
            save_figure(graph_median_final, f"figures/{suffix[1:]}/graph_median_{suffix[1:]}_median_final", center_median)
            plt.show()
        except:
            pass
        graph_random, index_random, center_random = parse_median_pickle_file(graph_random_file)
        graph_random = change_graph_attribute(graph_random)
        save_figure(graph_random, f"figures/{suffix[1:]}/graph_random_{suffix[1:]}_random", center_random)
        try:
            with open(final_random_file, "rb") as pkl:
                random_final = change_graph_attribute(pickle.load(pkl)[0])
            save_figure(random_final, f"figures/{suffix[1:]}/graph_random_{suffix[1:]}_random_final", center_random)
            plt.show()
        except:
            pass

