import csv
import os
import shutil
from collections import defaultdict
from typing import List, Tuple, Callable

import networkx as nx
import torch
from torch.utils.data import DataLoader

#from smilesparse.smiles import smiles_to_graph

BASE_PATH = "datasets/"

BASE_PATH_EDO = "../datasets/data_molecules/Data/"
SMILES_PATH = "molecular-data/odorants-smile.txt/odorants-smile.txt"
from multiprocessing import Pool
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.data import DataLoader as Dl, Data, Batch

from torch_geometric.data import InMemoryDataset, download_url, extract_zip
import numpy as np

Transform = Callable[[Data], Data]


class SMILESMolNet(InMemoryDataset):
    names = {
        'esol': ['ESOL', 'ESOL', 'delaney-processed', -1, -2],
        'freesolv': ['FreeSolv', 'FreeSolv', 'SAMPL', 1, 2],
        'lipo': ['Lipophilicity', 'lipophilicity', 'Lipophilicity', 2, 1],
        'pcba': ['PCBA', 'pcba', 'pcba', -1,
                 slice(0, 128)],
        'muv': ['MUV', 'muv', 'muv', -1, slice(0, 17)],
        'hiv': ['HIV', 'hiv', 'HIV', 0, -1],
        'bace': ['BACE', 'bace', 'bace', 0, 2],
        'BBBP': ['BBPB', 'BBBP', 'BBBP', -1, -2],
        'Tox21_AhR_training': ['Tox21', 'Tox21_AhR_training', 'Tox21_AhR_training', -1,
                  slice(0, 12)],
        'toxcast': ['ToxCast', 'toxcast', 'toxcast_data', 0,
                    slice(1, 618)],
        'sider': ['SIDER', 'sider', 'sider', 0,
                  slice(1, 28)],
        'clintox': ['ClinTox', 'clintox', 'clintox', 0,
                    slice(1, 3)],
        'hedonicity': ["Hedonicity", 'hedonicity', 'HEDO', 2, slice(3, None)]
    }
    url = "https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/molnet_publish/{}.zip"

    def __init__(self, name, root="datasets", transform=None, pre_transform=None, clean=False):
        self.name = name.lower()
        self.clean = clean

        assert self.name in self.names.keys()
        super(SMILESMolNet, self).__init__(root, transform, pre_transform)

        if self.name=="hedonicity":
            (self.data, self.slices), hed_val = torch.load(self.processed_paths[0])
        else :
            self.data, self.slices = torch.load(self.processed_paths[0])

        if type(self.transform) == HedonicityValueTransform:
            self.transform.hedVal = torch.tensor(hed_val)

    @property
    def raw_dir(self) -> str:
        name = 'raw{}'.format('_cleaned' if self.clean else '')
        return os.path.join(self.root, self.name, name )

    @property
    def processed_dir(self) -> str:
        name = 'processed{}'.format('_cleaned' if self.clean else '')
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        if self.name == "hedonicity":
            return ['{}.csv'.format(self.names[self.name][2]), '{}_score.csv'.format(self.names[self.name][2])]
        else:
            return '{}.csv'.format(self.names[self.name][2])

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        if self.name == "hedonicity":
            self.gen_raw_edo()
        else:
            url = self.url.format(self.names[self.name][1])
            path = download_url(url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def clear(self):
        shutil.rmtree(self.raw_dir)
        shutil.rmtree(self.processed_dir)

    def process(self):
        data_list = self.read_smiles()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data_list)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        if self.name=="hedonicity":
            hedo_scores = [float(el) for el in load_csv(self.raw_paths[1], delimiter=',')[0]]
            torch.save(((data, slices), hedo_scores), self.processed_paths[0])
        else :
            torch.save((data, slices), self.processed_paths[0])

    def gen_raw_edo(self):
        raw_dset, hedo_score = edo_raw(BASE_PATH_EDO)
        first_line = ['index', 'name','smile', 'attrs(74)']
        raw_dset = [first_line] + raw_dset

        with open(self.raw_paths[0], "w") as f:
            csvwriter = csv.writer(f, delimiter=',')
            [csvwriter.writerow(l) for l in raw_dset]


        with open(self.raw_paths[1], "w") as f:
            csvwriter = csv.writer(f, delimiter=',')
            csvwriter.writerow(hedo_score)

    def read_smiles(self) -> List[Data]:
        """
        reads a csv file with SMILES Formulas and labels and return graphs for torch_geometric
        :return:
        """
        dataset = load_csv(self.raw_paths[0], delimiter=',')[1:]

        labels = [el[self.names[self.name][4]] for el in dataset]
        labels = [ys if isinstance(ys, list) else [ys] for ys in labels]
        labels = [[float(y) if len(y) > 0 else float('NaN') for y in ys] for ys in labels]
        labels = [torch.tensor(ys, dtype=torch.float).view(1, -1) for ys in labels]

        smiles = [el[self.names[self.name][3]] + '\n' for el in dataset]

        with Pool(25) as p:
            graphs = p.map(smiles_to_graph, smiles)
        if self.clean:
            map = iso_map(graphs)
            graphs = [el for iso, el in zip(map, graphs) if iso]
            labels = [el for iso, el in zip(map, labels) if iso]

        # transforms nx_graph to torch_geometric
        data = [(gen_vector(g), gen_matrix(g), v) for g, v in zip(graphs, labels)]

        data = [(x, adj, attrs, y.view(-1)) for x, (adj,attrs), y in data]
        data = [Data(x=x, edge_index=adj, edge_attr=attrs, y=y.long()) for x, adj, attrs, y in data]

        return data
import networkx as nx
from tqdm import tqdm
def iso_map(graphs):
    iso_map = [1 for _ in graphs]
    node_match = lambda a,b: a["symbol"] == b["symbol"]
    edge_match= lambda a,b:  a["bond"] == b["bond"]
    for i, g1 in enumerate(tqdm(graphs)):
        if iso_map[i]:
            for j, g2 in enumerate(graphs):
                if i!=j and nx.is_isomorphic(g1,g2, node_match=node_match, edge_match=edge_match):
                    iso_map[i] = 0
                    iso_map[j] = 0
    return iso_map
    #return [el for iso, el in zip(indexes, graphs) if iso]

def gen_matrix(graph: nx.Graph) -> torch.tensor:
    """
    generate adjency matrix
    TODO : Special edges
    :param graph:
    :return:
    """
    n_edge_type = 1
    adjacency_matrix = torch.zeros(graph.number_of_nodes(), graph.number_of_nodes(), n_edge_type)
    edge_value = {'': 1, '/': 1.5, '\\': 1.5, "-": 1, "=": 2, "#": 3, ".": 0}
    for x, y in graph.edges():
        adjacency_matrix[x - 1, y - 1, 0] = edge_value[graph.edges[(x, y)]["bond"]] != 0
    adjacency_matrix += adjacency_matrix.transpose(0, 1)
    adjacency_matrix = adjacency_matrix.view((graph.number_of_nodes(), graph.number_of_nodes()))
    sparse = dense_to_sparse(adjacency_matrix)
    #edge_attr = [adjacency_matrix[x,y] for x,y in sparse]
    return sparse


atoms = ["C", "N", "O", "S", "P", "BR", "B", "F", "CL", "I", "H", "NA", "CA"]
indexes = {a: i for i, a in enumerate(atoms)}


def gen_vector(molecule: nx.graph) -> torch.tensor:
    """
    TODO :special parameters
    generates One Hot encoding
    :param molecule:
    :return:
    """
    ret = torch.zeros(molecule.number_of_nodes(), len(atoms))

    for i, n in enumerate(molecule.nodes()):
        ret[i][indexes[molecule.nodes[n]["symbol"].upper()]] = 1
    return ret


def molnet_dataloader(name: str, transform: Transform, ratio, clean=False, **kwargs) -> Tuple[DataLoader, DataLoader]:
    dataset = SMILESMolNet(name, transform=transform, clean=clean)

    l1 = int(len(dataset) * ratio)
    lens = [l1, len(dataset) - l1]
    train, test = torch.utils.data.random_split(dataset, lens)

    init_fn = lambda x: np.random.seed(kwargs["seed"])

    return \
        Dl(train, batch_size=kwargs["batch_size"], shuffle=True, worker_init_fn=init_fn ), \
        Dl(test, batch_size=kwargs["batch_size"], shuffle=True, worker_init_fn=init_fn )


def edo_raw(path: str) -> Tuple[List[List], List[float]]:
    with open(os.path.join(path, "molecular-data/odorants-smile.txt/odorants-smile.txt"), "r") as f:
        smiles = {int(i): s[:-1] for i, s in map(lambda l: l.split("\t"), f)}
    association_map = load_csv(os.path.join(path, "odorants.csv"))
    inv_map = {int(el[1]): int(el[0]) for el in association_map}
    mols = {int(el[1]): el[2] for el in association_map}

    csv_array = load_csv(os.path.join(path, "qualities.csv"), delimiter="\t")

    classes = {el: int(el) for el in {line[1] for line in csv_array}}
    hedonicity_score = [0 for _ in classes]

    labels = defaultdict(lambda: [0 for _ in classes])
    for line in csv_array:
        cl = classes[line[1]] - 1
        labels[int(line[0])][cl] += 1

        hedonicity_score[int(line[1]) - 1] = float(line[3])

    graph_labels = [[i, mols[k], smiles[k]] + labels[inv_map[k]] for i, k in enumerate(inv_map.keys())]
    return graph_labels, hedonicity_score


def load_csv(file: str, delimiter: str = ';') -> List[List[str]]:
    out = list()
    with open(file, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=delimiter)
        for row in data:
            out.append(row)
    return out


class BinClassTransform(object):
    def __init__(self, param):
        self.param = param

    def __call__(self, data: Data) -> Data:
        data.y = int(data.y[self.param])
        return data


class HedonicityValueTransform(object):
    def __init__(self, hed_val=torch.zeros(74)):
        self.hedVal = hed_val

    def __call__(self, data: Data) -> Data:
        val = (self.hedVal * data.y) / data.y.sum()
        data.y = val
        return data


class MultiLabelTransform(object):
    def __init__(self, labels: List[int]):
        self.labels = labels

    def __call__(self, data: Data) -> Data:
        data.y = torch.tensor([int(data.y[el]) for el in self.labels])
        return data


class ChangeDim(object):
    def __init__(self):
        self.size = 50

    def __call__(self, data):
        data.x = data.x[:, :self.size]
        return data


def tox_21_dl(directory, train, test, **kwargs):
    train = TUDataset(directory, train, transform=ChangeDim())
    test = TUDataset(directory, test, transform=ChangeDim())

    init_fn = lambda x: np.random.seed(kwargs["seed"])

    return \
        Dl(train, batch_size=kwargs["batch_size"], shuffle=True, worker_init_fn=init_fn ), \
        Dl(test, batch_size=kwargs["batch_size"], shuffle=True, worker_init_fn=init_fn )

def aids_dl(directory, ratio, **kwargs):
    dataset = TUDataset(directory, "AIDS", transform=ChangeDim())
    l1 = int(len(dataset) * ratio)
    lens = [l1, len(dataset) - l1]
    train, test = torch.utils.data.random_split(dataset, lens)

    init_fn = lambda x: np.random.seed(kwargs["seed"])
    return \
        Dl(train, batch_size=kwargs["batch_size"], shuffle=True, worker_init_fn=init_fn ), \
        Dl(test, batch_size=kwargs["batch_size"], shuffle=True,worker_init_fn=init_fn)

def get_ratio(dataloader:Dl):
    return np.array([el.y.numpy() for el in Dl(dataloader.dataset, batch_size=1,shuffle=False)]).mean()

class transform_output(object):

    def __init__(self, function):
        self.fun= function

    def __call__(self, data):
        y = self.fun(Batch.from_data_list([data]))
        data.y = y
        return data

from torch_geometric.data import Dataset

def transform_dataset_labels(dataloader, function):
    #dl = Dl(dataloader.dataset, batch_size=1,shuffle=False)

    tr = transform_output(function)
    dl= Dl(dataloader.dataset, batch_size=1,shuffle=False)
    dl.dataset.dataset.transform=tr
    return dl


class autoencoder_transform(object):
    def __init__(self):
        pass #self.fun= function

    def __call__(self, data):
        data.y = None, data.y
        return data


class autoencoder_reverse_transform(object):
    def __init__(self):
        pass  # self.fun= function

    def __call__(self, data):
        data.y = data.y[1]
        return data
from utils import graph_generator
from utils import featgen
from utils.synthetic_structsim import *

def preprocess_input_graph(G, labels, normalize_adj=False):
    """ Load an existing graph to be converted for the experiments.
    Args:
        G: Networkx graph to be loaded.
        labels: Associated node labels.
        normalize_adj: Should the method return a normalized adjacency matrix.
    Returns:
        A dictionary containing adjacency, node features and labels
    """
    adj = np.array(nx.to_numpy_matrix(G))
    if normalize_adj:
        sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
        adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)

    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].shape[0]
    f = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        f[i, :] = G.nodes[u]["feat"]

    # add batch dim
    adj = np.expand_dims(adj, axis=0)
    f = np.expand_dims(f, axis=0)
    labels = np.expand_dims(labels, axis=0)
    return {"adj": adj, "feat": f, "labels": labels}

def synth_dl(size):
    train = clique(size//2, size, **kwarsg)
    test = clique(size//2, size, **kwarsg)
    #data, slices = InMemoryDataset.collate(train)

    l1 = int(len(data_list) * ratio)
    lens = [l1, len(data_list) - l1]
    init_fn = lambda x: np.random.seed(kwargs["seed"])

    #DD = Dl((data,slices), batch_size=1, shuffle=True, worker_init_fn=init_fn)
    #train, test = torch.utils.data.random_split(DD.dataset, lens)
    #train, test = data, data

    return \
        Dl(data_list, batch_size=kwargs["batch_size"], shuffle=True, worker_init_fn=init_fn ), \
        Dl(data_list, batch_size=kwargs["batch_size"], shuffle=True, worker_init_fn=init_fn )
"""
def clique(positive, size,  **kwargs):
    type = "ba"
    positives = [build_graph(np.random.randint(20,50), "ba", [["house"]], start=0, m=5)[0] for _ in range(positive)]
    negatives = [eval(type)(0, np.random.randint(20,50))[0] for _ in range(size-positive)]
    graphs = positives+negatives
    fg = featgen.ConstFeatureGen(np.ones(2, dtype=float))
    [fg.gen_node_features(el)for el in graphs]
    data_list = list()
    for i, el in enumerate(graphs):
        data = preprocess_input_graph(el)
        adj = dense_to_sparse(torch.tensor(data["adj"], dtype=torch.float)[0])
        x = torch.tensor(data["feat"], dtype=torch.float)[0]
        data_list.append(Data(x=x, edge_index= adj[0],edge_attr=adj[1], y = torch.tensor([i<positive]).long()))
    return data_list
"""
def synth_dataloader(params, positive,ratio, **kwargs):
    dataset = CliqueDset("syn1_"+str(params["shape"])+"m_"+str(params["m"]), positive, params)


    l1 = int(len(dataset) * ratio)
    lens = [l1, len(dataset) - l1]
    train, test = torch.utils.data.random_split(dataset, lens)

    init_fn = lambda x: np.random.seed(kwargs["seed"])

    return \
        Dl(train, batch_size=kwargs["batch_size"], shuffle=True, worker_init_fn=init_fn), \
        Dl(test, batch_size=kwargs["batch_size"], shuffle=True, worker_init_fn=init_fn)

class CliqueDset(InMemoryDataset):
    def __init__(self, name, positive, params,  root="datasets", transform=None, pre_transform=None, clean=False):
        self.params = params
        self.name = name.lower()
        self.clean = clean
        self.positive= positive
        self.type= params["shape"][0]

        super(CliqueDset, self).__init__(root, transform, pre_transform)


        self.data, self.slices = torch.load(self.processed_paths[0])

        if type(self.transform) == HedonicityValueTransform:
            self.transform.hedVal = torch.tensor(hed_val)

    @property
    def raw_dir(self) -> str:
        name = 'raw{}'.format('_cleaned' if self.clean else '')
        return os.path.join(self.root, self.name, name )

    @property
    def processed_dir(self) -> str:
        name = 'processed{}'.format('_cleaned' if self.clean else '')
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        return '{}.csv'.format(self.name)

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass
        """
        if self.name == "hedonicity":
            self.gen_raw_edo()
        else:
            url = self.url.format(self.names[self.name][1])
            path = download_url(url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)
        """
    def clear(self):
        shutil.rmtree(self.raw_dir)
        shutil.rmtree(self.processed_dir)

    def process(self):
        data_list = self.clique(self.positive, 2*self.positive)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data_list)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])

    def clique(self, positive, size):
        random_shape = self.params["rdm_params"]
        positives = [build_graph(np.random.randint(*random_shape), self.params["shape"][0], [self.params["shape"][1]], start=0, m=self.params["m"],rdm_basis_plugins=self.params["rdm_basis"]) for _ in
                     range(positive)]
        if self.params["shape"][0] =="ba":
            negatives = [eval(self.params["shape"][0])(0, np.random.randint(*random_shape), m=self.params["m"]) for _ in range(size - positive)]
        else :
            negatives = [eval(self.params["shape"][0])(0, np.random.randint(*random_shape)) for _ in range(size - positive)]

        graphs = positives + negatives
        graphs, indexes= list(zip(*graphs))
        #graphs = perturb(graphs, 0.01)
        fg = featgen.GaussianFeatureGen(np.ones([2]), np.ones([2]))#
        #fg = featgen.ConstFeatureGen(np.array([1,2]))
        [fg.gen_node_features(el) for el in graphs]
        data_list = list()
        for i, el in enumerate(graphs):
            #data = preprocess_input_graph(el)
            #adj= dense_to_sparse(torch.tensor(data["adj"][0]))
            #torch.tensor(data["feat"], dtype=torch.float32)

            adj=get_adj(el)
            x = torch.tensor(get_feat(el)[0], dtype=torch.float32)#torch.tensor(data["feat"], dtype=torch.float32) #
            data_list.append(Data(x=x, edge_index=adj[0], edge_attr=adj[1], y=torch.tensor([i < positive]).long(),  z= indexes[i]))
        return data_list


def get_adj(graph: nx.Graph) -> torch.tensor:
    """
    generate adjency matrix
    TODO : Special edges
    :param graph:
    :return:
    """
    n_edge_type = 1
    adjacency_matrix = torch.zeros(graph.number_of_nodes(), graph.number_of_nodes())
    for x, y in graph.edges():
        adjacency_matrix[x, y] = 1
    adjacency_matrix += adjacency_matrix.transpose(0, 1)
    adjacency_matrix = adjacency_matrix.view((graph.number_of_nodes(), graph.number_of_nodes()))
    sparse = dense_to_sparse(adjacency_matrix)
    #edge_attr = [adjacency_matrix[x,y] for x,y in sparse]
    return sparse

def get_feat(G) -> torch.tensor:
    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].shape[0]
    f = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        f[i, :] = G.nodes[u]["feat"]

    # add batch dim
    f = np.expand_dims(f, axis=0)
    return f

def perturb(graph_list, p):
    """ Perturb the list of (sparse) graphs by adding/removing edges.
    Args:
        p: proportion of added edges based on current number of edges.
    Returns:
        A list of graphs that are perturbed from the original graphs.
    """
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list