import json
import pickle

import code
import jraph
import numpy
import numpy as np
import pandas
import pandas as pd
from multiprocessing import Pool
from functools import partial
from jax import numpy as jnp
import jax
from transformers import BertTokenizer

from GNN_Transformer.loader import ProtBERTDataset, ProtBERTCollatePrecomputeBERT_CLS
from GNN_Transformer.make_predict import make_apply_bert


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (jax.numpy.integer, numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (jax.numpy.floating, numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray, jax.numpy.DeviceArray)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def build_intervals(graph_indices, probs):
    intervals = []
    start = 0
    current_ind = graph_indices[0]
    for i in range(probs.shape[0]):
        if current_ind != graph_indices[i]:
            intervals.append((start, i))
            start, current_ind = i, graph_indices[i]
    intervals.append((start, probs.shape[0]))
    intervals = np.asarray(intervals)
    return intervals


def compute_SI_double(probs, data, pattern, graph_indices):
    positive_indices = list(filter(lambda index: pattern[index] == 1, pattern.keys()))
    negative_indices = list(filter(lambda index: pattern[index] == 0, pattern.keys()))
    if not len(negative_indices):
        return compute_SI_single(probs, data, pattern, graph_indices)
    DI = 0.6 * (len(positive_indices) + len(negative_indices)) + 1
    node_support_indices = np.where(
        ~np.any(data[:, negative_indices], axis=1) & np.all(data[:, positive_indices], axis=1))
    if not len(node_support_indices[0]):
        return 0
    probs = probs[node_support_indices]
    positive_probs = probs[:, positive_indices]
    f = np.vectorize(lambda x: np.log(x) if x else 0)
    negative_probs = f(1 - np.exp(probs[:, negative_indices]))
    graph_indices = graph_indices[node_support_indices]

    res = 0
    positive_sums = np.apply_along_axis(np.sum, 1, positive_probs)
    negative_sums = np.apply_along_axis(np.sum, 1, negative_probs)
    sums = positive_sums + negative_sums
    intervals = build_intervals(graph_indices, probs)
    for start, end in intervals:
        res -= np.min(sums[start: end])
    return res / DI


def compute_SI_single(probs, data, pattern, graph_indices):
    # labels is the array in which the class of each node is stored
    # target is an integer that indicates which class we are interested in computing
    # probs contains P(component c = 1 for node v)
    positive_indices = list(filter(lambda index: pattern[index] == 1, pattern.keys()))
    DI = 0.6 * len(positive_indices) + 1
    node_support_indices = np.where(np.all(data[:, positive_indices], axis=1))
    if not len(node_support_indices[0]):
        return 0
    probs = probs[node_support_indices]
    graph_indices = graph_indices[node_support_indices]

    probs = probs[:, positive_indices]
    intervals = build_intervals(graph_indices, probs)
    res = 0
    sums = np.apply_along_axis(np.sum, 1, probs)
    for start, end in intervals:
        res -= np.min(sums[start: end])
    return res / DI


def build_graph_indices(embedding_list):
    graph_indices = np.concatenate(list(map(lambda x: list(map(len, x)), embedding_list)))
    return np.repeat(np.arange(len(graph_indices)), graph_indices)


def build_embedding_matrix(embedding_list):
    embeddings = list(map(np.concatenate, embedding_list))
    return np.concatenate(embeddings)


def f(x, layer):
    def g(y):
        return y.nodes

    return list(map(g, jraph.unbatch(x['intermediates']['graphs'][layer])))[::2]


def build_embeddings_and_graph_indices_from_intermediates(intermediates, layer=0):
    pool = Pool()
    embeddings = pool.map(partial(f, layer=layer), intermediates)
    pool.close()
    # embeddings = list(map(f, intermediates))
    graph_indices = build_graph_indices(embeddings)
    embeddings = build_embedding_matrix(embeddings)
    return embeddings, graph_indices


def get_predictions_from_file(file, graph_indices):
    df = pd.read_csv(file, delimiter=";")
    predictions = df["pred"].to_numpy()
    predictions = np.vectorize(lambda x: x >= 0.5)(predictions)
    print("plus:")
    print(np.sum(predictions))
    _, graph_sizes = np.unique(graph_indices, return_counts=True)
    predictions = np.repeat(predictions, graph_sizes)
    return predictions


def build_dataset_from_file(intermediate_file, predication_file, layer=0):
    # returns an activation matrix, label, graph_indices
    with open(intermediate_file, "rb") as pkl:
        intermediates = pickle.load(pkl)
    activation_matrix, graph_indices = build_embeddings_and_graph_indices_from_intermediates(intermediates, layer)
    min_component = np.apply_along_axis(np.min, 0, activation_matrix)
    max_component = np.apply_along_axis(np.max, 0, activation_matrix)
    #f = np.vectorize(lambda x: np.maximum(np.sign(x), 0.0))
    activation_matrix = np.sign((activation_matrix - min_component)/(max_component - min_component)-0.5)
    labels = get_predictions_from_file(predication_file, graph_indices)
    activation_matrix = pandas.DataFrame(data=activation_matrix,
                                         columns=[f"l_{layer}c_{i}" for i in range(activation_matrix.shape[1])])
    with open(f"refrence vectores {layer}", 'wb') as pkl:
        pickle.dump((min_component, max_component), pkl)
    return activation_matrix, graph_indices, labels


def convert_subgroup_to_dict(subgroup):
    components = subgroup.split("AND")
    try:
        components = {comp.split("==")[0]: int(float(comp.split("==")[1].strip())) for comp in components}
    except:
        return None
    components = {int(key[key.index('c') + 2:]): value for key, value in components.items()}
    return components


def read_from_beam_search_files(file_name):
    df = pd.read_csv(file_name, delimiter=",")
    df["Rule"] = df["Rule"].apply(convert_subgroup_to_dict)
    return df[df.Rule.notnull()]


def convert_conjunction_to_dict(conjunction):
    components_dict = {int(x.attribute_name[x.attribute_name.index("c") + 2:]): int(x.attribute_value) for x in
                       conjunction.selectors}
    return components_dict


def get_support_indices(rule, activation_matrix, graph_indices):
    if not isinstance(rule, dict):
        components = list(convert_conjunction_to_dict(rule[1]).keys())
    else:
        components = list(rule.keys())
    return np.unique(graph_indices[np.where(np.all(activation_matrix[:, components], axis=1))])


def convert_jraph_to_numpy_tuple(graph):
    return np.array(graph[0].nodes), np.array([graph[0].senders, graph[0].receivers]), graph[0].edges


def write_support_graphs_to_file(rule, activation_matrix, graph_indices, graphs_data, file_to_write):
    print(rule)
    graphs = graphs_data['_graphs']
    pattern = rule.Rule
    rule_support_indices = get_support_indices(pattern, activation_matrix, graph_indices)
    rule_support_graphs = graphs.iloc[rule_support_indices]
    rule_support_graphs = rule_support_graphs.apply(convert_jraph_to_numpy_tuple)
    list_to_write = [(rule_support_indices[i],) + item for i, item in
                     enumerate(rule_support_graphs.iteritems())]
    list_to_write = list(map(lambda x: (x[0], x[1:][1]), list_to_write))
    key = int(bool(rule.target.split("==")[-1]))
    if False:
        print("_____________________________Debug________________________________________")
        for i, l in enumerate(list_to_write):
            if len(l) != 3:
                print(f"index: {i}, len: {len(l)}\n {l}")
                print(rule_support_graphs.iloc[[i]])
                break
        code.interact(local=locals())
    with open(file_to_write, "w") as file:
        json.dump({key: list_to_write}, file, cls=NumpyEncoder)


def load_mol_dataset(csv_file, seq_file, bert_file, seq_col="seq_id", mol_col="_SMILES", label_col="pred"):
    dataset_class = ProtBERTDataset(csv_file, seq_col, mol_col, label_col,
                                    atom_features=["AtomicNum", "ChiralTag", "Hybridization", "FormalCharge",
                                                   "NumImplicitHs", "ExplicitValence", "Mass", "IsAromatic"],
                                    bond_features=["BondType", "Stereo", "IsAromatic"])
    import tables
    h5file = tables.open_file(bert_file, mode='r', title="TapeBERT")
    bert_table = h5file.root.bert.BERTtable

    collate = ProtBERTCollatePrecomputeBERT_CLS(bert_table,
                                                padding_n_node=32,
                                                padding_n_edge=64,
                                                n_partitions=0,
                                                from_disk=False,
                                                line_graph=False)
    make_collate = collate.make_collate()
    if False:
        code.interact(local=locals())
        print("exited.")
    return dataset_class.data, make_collate


def filter_by_seq_id(dataset: pd.DataFrame, activation_matrix: pd.DataFrame, graph_indices: np.array, labels: np.array,
                     seq_id):
    # code.interact(local=locals())
    indices = dataset.index[dataset['seq_id'] == seq_id]
    dataset = dataset.iloc[indices]
    rows_to_keep = np.in1d(graph_indices, indices)
    activation_matrix = activation_matrix.iloc[rows_to_keep]
    labels = labels[rows_to_keep]
    _, graph_indices = np.unique(graph_indices[rows_to_keep], return_inverse=True)
    return dataset, activation_matrix, graph_indices, labels

def find_activated_nodes(index, activation_matrix, graph_indices, rule):
    """
    :param index: index of the graph in dataset
    :param activation_matrix: activation matrix
    :param graph_indices: An array to hold graph index of each row of the activation matrix
    :param rule: Activation pattern
    :return:
    """
    rows = activation_matrix[graph_indices == index]
    components = list(rule.keys())
    activated_nodes = np.where(np.all(rows[:, components], axis=1))
    return activated_nodes[0]

