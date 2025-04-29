import numpy as np
import functools
from tqdm import tqdm
import torch
from pyemd import emd
import clustering
from metrics import au_roc, au_pr, f1_score, optimal_threshold
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool
import os

def get_graph_list(data):
    graphs = clustering.molecular_maping(data)
    graph_list = [0 for _ in range(len(graphs))]
    for i, el in graphs.items():
        graph_list[i] = el
    return graph_list


def word_moover_distance_experiment(train_data, test_data, file_tag="", caption="", ** kwargs):
    out = list()
    ds=list()
    for layer in range(len(train_data)):
        train_graph_list = get_graph_list(train_data[layer])
        test_graph_list = get_graph_list(test_data[layer])

        distances = get_distances(graph_distances_matrix, train_graph_list, test_graph_list,
                                  file_tag=str(layer) + "_" + file_tag, medhod_name="gmd", **kwargs)
        ds.append(distances)

        model_name = "emd_"+ str(layer)
        v, ths = run_knn(distances, file_tag, model_name, **kwargs)
        kwargs.update({"threshold_"+ model_name+str(2**k):ths[k] for k in range(8)})
        out.append(v)
    print(print_res_to_latex(out,list(range(len(train_data))) ,[2 ** k for k in range(8)], caption, metrics=kwargs["knn_metrics"]))
    return out


def gmd_AE_distance_experiment(train_data, test_data, file_tag="", caption="", ** kwargs):
    train_graph_list = get_graph_list(train_data)
    test_graph_list = get_graph_list(test_data)

    distances = get_distances(graph_distances_matrix, train_graph_list, test_graph_list,
                                  file_tag= file_tag, medhod_name="gmd_ae", **kwargs)

    model_name = "gmd_AE_"
    vals, ths = run_knn(distances, file_tag, model_name, **kwargs)
    print(print_res_to_latex([vals], layers=[0], layer_size=[2 ** k for k in range(8)], caption=caption, metrics=kwargs["knn_metrics"]))
    kwargs.update({"threshold_" + model_name + str(2 ** k): ths[k] for k in range(8)})

    return vals
def graph_distances_matrix(train, test, **kwargs):
    pool = Pool(23)
    traindata = [{tuple(v[0][0].tolist()) for v in el} for el in train]
    testdata = [{tuple(v[0][0].tolist()) for v in el} for el in test]

    distances = np.zeros((len(test), len(train)))

    for i, g1 in enumerate(tqdm(testdata)):
        parital_gmd = functools.partial(gmd, graph2=g1)
        d = pool.map(parital_gmd, traindata, chunksize=5)
        distances[i,:] = np.array(d)

    return distances

def gmd(graph1, graph2):
    """
    graph1 and 2 ar list of array (of node embedding)
    :param graph1:
    :param graph2:
    :param distance:
    :return:
    """

    graphset = graph1.union(graph2)
    graph_dict = {k: i for i, k in enumerate(graphset)}

    distance_matrix = np.zeros((len(graph_dict),len(graph_dict)))

    graph_list1 = [(v, -1, i) for v, i in graph_dict.items() if v in graph1]
    graph_list2 = [(v, -1, i) for v, i in graph_dict.items() if v in graph2]
    for v1, id1, i in graph_list1: #graph_dict.items():
        for v2, id2, j in graph_list2:
            if distance_matrix[i, j] == 0.0:
                distance_matrix[i, j] = distance_matrix[j, i] = np.linalg.norm(np.array(v1)-np.array(v2))#mahalanobis(v1,v2, matrix)<

    def frequencies(graph):
        d = np.zeros(len(graph_dict))
        graph_size =len(graph)
        for el in graph:
            d[graph_dict[el]]+=1
        return d/float(graph_size)

    d1 = frequencies(graph1)
    d2 = frequencies(graph2)
    e = emd(d1, d2, distance_matrix)

    return e
from torch_geometric.data import DataLoader as Dl

def graph_edit_distance(file_tag="", caption="", ** kwargs):
    train_graphs = dataset_to_nx(kwargs["trainset"])
    test_graphs = dataset_to_nx(kwargs[file_tag])

    distances = get_distances(ged_distance_matrix, train_graphs, test_graphs, file_tag=file_tag, medhod_name="ged",
                              **kwargs)

    model_name = "ged_"
    vals, ths = run_knn(distances, file_tag, model_name, **kwargs)
    print(print_res_to_latex([vals], layers=[0], layer_size=[2 ** k for k in range(8)], caption=caption, metrics=kwargs["knn_metrics"]))
    kwargs.update({"threshold_" + model_name + str(2 ** k): ths[k] for k in range(8)})

    return vals


def ged_distance_matrix(train, test, **kwargs):
    nb2s = [[nx.Graph(g2.subgraph(list(g2.adj[node]) + [node])) for node in g2.nodes] for g2 in train]

    distances = np.zeros((len(test), len(train)))
    pool = Pool(23)

    train_els = list(zip(train, nb2s))
    for i, g1 in enumerate(tqdm(test)):
        test_el = (g1, [nx.Graph(g1.subgraph(list(g1.adj[node]) + [node])) for node in g1.nodes])
        partialged = functools.partial(ged, g2=test_el)
        d = pool.map(partialged, train_els, chunksize=5)
        distances[i, :] = np.array(d)
    return distances
    
def my_node_match(x, y):
    return x["x"] == y["x"]
def my_edge_match(x, y):
    return x["edge_attr"] == y["edge_attr"]

def ged(g1, g2):
    if abs(len(g1[0]) - len(g2[0]))>11:
        return 150
    if len(g1[0]) < len(g2[0]):
        g1, g2 = g2, g1
    g1, nb1s = g1
    g2, nb2s = g2

    n = len(g1)
    cost = np.ones((n,n))
    for i, nb1 in zip(g1.nodes, nb1s):
        for j,nb2 in zip(g2.nodes, nb2s):
            cost[i,j]= nx.graph_edit_distance(nb1,nb2, node_match=my_node_match, edge_match=my_edge_match, roots =(i,j))
            if  cost[i,j]!= cost[i,j]: # if nan because i,j are fixed it is impossible somitmes
                cost[i,j] = nx.graph_edit_distance(nb1,nb2, node_match=my_node_match, edge_match=my_edge_match)
    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except Exception:
        print(cost)
    return cost[row_ind, col_ind].sum()


def gnn_ged_experiment(train_data, test_data, file_tag="", caption="", **kwargs):
    out = list()
    ds=list()
    for layer in range(len(train_data)):
        train_graph_list = get_graph_list(train_data[layer])
        test_graph_list = get_graph_list(test_data[layer])

        distances = get_distances(hungarian_distance_matrix, train_graph_list, test_graph_list,
                                  file_tag=str(layer) + "_" + file_tag, medhod_name="hungarian", **kwargs)
        ds.append(distances)

        model_name = "hungarian_"+ str(layer)
        v, ths = run_knn(distances, file_tag, model_name, **kwargs)
        kwargs.update({"threshold_"+ model_name+str(2**k):ths[k] for k in range(8)})
        out.append(v)
    print(print_res_to_latex(out,list(range(len(train_data))), [2 ** k for k in range(8)], caption,metrics=kwargs["knn_metrics"]))
    return out



def hungarian_distance_matrix(train, test, **kwargs):
    distances = np.zeros((len(test), len(train)))

    pool = Pool(23)
    train_dl = ([np.array([n[0][0].numpy() for n in el]) for el in train])
    test_dl = ([np.array([n[0][0].numpy() for n in el]) for el in test])

    for i, g1 in enumerate(tqdm(test_dl)):
        #test_el =(g1,[nx.Graph(g1.subgraph(list(g1.adj[node])+[node])) for node in g1.nodes])
        partial_hung = functools.partial(
            gnn_hungarian_algorithm, g2=g1, neutral=None, norm=None)
        d = pool.map(partial_hung, train_dl, chunksize=5)
        distances[i,:] = np.array(d)
    return distances


def ged_AE_distance_experiment(train_data, test_data, file_tag="", caption="", ** kwargs):

    train_graph_list = get_graph_list(train_data)
    test_graph_list = get_graph_list(test_data)

    distances = get_distances(hungarian_distance_matrix, train_graph_list, test_graph_list,
                                  file_tag= file_tag, medhod_name="ged_ae", **kwargs)

    model_name = "ged_AE_"
    vals, ths = run_knn(distances, file_tag, model_name, **kwargs)
    print(print_res_to_latex([vals], layers=[0], layer_size=[2 ** k for k in range(8)], caption=caption, metrics=kwargs["knn_metrics"]))
    kwargs.update({"threshold_" + model_name + str(2 ** k): ths[k] for k in range(8)})

    return vals

def gnn_hungarian_algorithm(g1,g2, neutral=None,norm=None):
    norm = norm or (lambda a,b: np.linalg.norm(b-a)) # default value
    neutral = neutral or np.zeros(len(g1[0]))
    if len(g1) < len(g2):
        g1, g2 = g2, g1
    if len(g1) != len(g2):
        g2= np.concatenate([g2, np.array([neutral for _ in range(len(g1)- len(g2))])])

    cost = np.ones((len(g1),len(g2)))
    for i, iel in enumerate(g1):
        for j, jel in enumerate(g2):
            cost[i,j]= norm(iel, jel)##torch.dist(,b,2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind].sum()


def jaccard_distance_experiment(file_tag="", caption="", ** kwargs):
    train_graphs = list(map(nx_to_set, dataset_to_nx(kwargs["trainset"])))
    test_graphs = list(map(nx_to_set,dataset_to_nx(kwargs[file_tag])))

    distances = get_distances(jaccard_distance_matrix, train_graphs, test_graphs, file_tag=file_tag,
                              medhod_name="jaccard", **kwargs)

    model_name = "jaccard"
    vals, ths = run_knn(distances, file_tag, model_name, **kwargs)
    print(print_res_to_latex([vals], layers=[0],layer_size=[2 ** k for k in range(8)], caption=caption,metrics=kwargs["knn_metrics"]))
    kwargs.update({"threshold_"+ model_name+str(2**k):ths[k] for k in range(8)})
    return vals

def dataset_to_nx(dset:Dl):
    ds = Dl(dset.dataset, batch_size=1,shuffle=False)
    return [to_networkx(el,node_attrs=['x'],edge_attrs=["edge_attr"]) for el in ds]

def nx_to_set(g):
    node_attr = [np.argmax(g.nodes[el]["x"])for el in g]
    edge_attr = [(np.argmax(g.edges[a,b]["edge_attr"]), np.argmax(g.nodes[a]["x"]), np.argmax(g.nodes[b]["x"]))
                 for a, b in g.edges]
    edge_attr = [(n, min(a,b), max(a,b) )for n, a,b in edge_attr]
    return set(node_attr+edge_attr)

def jaccard_distance_matrix(train, test, **kwargs):
    distances = np.zeros((len(test), len(train)))
    pool = Pool(23)
    for i, g1 in enumerate(tqdm(test)):
        partial_jaccard = functools.partial(jaccard_distance, g2=g1)
        d = pool.map(partial_jaccard, train, chunksize=5)
        distances[i, :] = np.array(d)
    return distances

def jaccard_distance(g1,g2):
    union = len(set.union(g1,g2))
    intersection = len(set.intersection(g1,g2))
    return (union+intersection)/union

def get_distances(distance_fn, train, test, reset_storage=False, file_tag="", medhod_name="", **kwargs):
    dirname = os.path.join("distances", medhod_name)
    sub_dirname = os.path.join(dirname, kwargs["new_data"])

    file_name =os.path.join(sub_dirname, medhod_name + "_" + file_tag + ".pth")
    if reset_storage or not os.path.exists(file_name):
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        if not os.path.exists(sub_dirname):
            os.mkdir(sub_dirname)

        distances = distance_fn(train,test, **kwargs)
        torch.save(distances, file_name)
    return torch.load(file_name)

def get_distance_matrix(file_tag="", medhod_name="", **kwargs):
    dirname = os.path.join("distances", medhod_name)
    sub_dirname = os.path.join(dirname, kwargs["new_data"])

    file_name =os.path.join(sub_dirname, medhod_name + "_" + file_tag + ".pth")

    return torch.load(file_name)




def run_knn(distances, test_name, model_name, **kwargs):
    vals =list()
    ths = list()

    train_classes = np.concatenate([el.y for el in kwargs["trainset"]])
    test_classes = np.concatenate([el.y for el in kwargs[test_name]])

    if "threshold" in kwargs:
        train_classes = train_classes > kwargs["threshold"]
        test_classes = test_classes > kwargs["threshold"]
    #cumulative_match_caracteristic(distances, train_classes, test_classes, model_name, **kwargs)
    for i in range(8):
        val, th = knn(distances, train_classes, test_classes, 2**i, model_name, **kwargs)
        vals.append([float("{:0.4}".format(float(v)))for v in val])
        ths.append(th)
    return vals, ths

import matplotlib.pyplot as plt

def cumulative_match_caracteristic(distance_matrix, classes, neg=0):
    if distance_matrix.shape[0]==distance_matrix.shape[1]:
        distance_matrix+= np.identity(distance_matrix.shape[0])*1.0e99

    #bests = distance_matrix.argpartition(k)[:,0:k]
    bests = distance_matrix.argsort()

    cl = np.vectorize(lambda x : classes[x])
    clmatrix = cl(bests)
    ranks= [0 for _ in range(distance_matrix.shape[0]) ]
    for i, l in enumerate(bests):
        k = 0
        while k < len(l) and clmatrix[i,k] != (neg != classes[i]) :
            k+=1
        ranks[k]+=1
    ranks = list(np.cumsum(ranks)/distance_matrix.shape[0])

    return ranks
    #tb.plot_cmc(ranks, model_name, **kwargs)
    #plt.plot(ranks[:100] )
    #plt.show()


    #target = np.array(test_classes)




def knn(distance_matrix, classes, test_classes, k, model_name, **kwargs):
    """
    computes k nearest neibors, and returns
    :param distance_matrix:
    :param classes:
    :param test_classes:
    :param k:
    :return:
    """
    if distance_matrix.shape[0]==distance_matrix.shape[1]:
        distance_matrix+= np.identity(distance_matrix.shape[0])*1.0e99
    k = min(distance_matrix.shape[1]-1,k)
    #bests = distance_matrix.argpartition(k)[:,0:k]
    bests = distance_matrix.argsort()

    cl = np.vectorize(lambda x : classes[x])
    clmatrix = cl(bests)
    pred = np.zeros(len(test_classes))
    for i, l in enumerate(bests):
        k2=k
        while k2+1 < len(l) and distance_matrix[i,l[k2+1]] == distance_matrix[i,l[k2]]:
            k2+=1
        pred[i] = clmatrix[i,0:k2].mean()

    target = np.array(test_classes)

    if "threshold_" + model_name not in kwargs:
        th = optimal_threshold(pred, target)
    else :
        th = kwargs["threshold_" + model_name+str(k)]
    if th==1:
        th -= 1e-10
    metric_functions =dict()
    metric_functions["aur"]= lambda p, t : au_roc(np.array([1-p,p]).transpose(), t)

    metric_functions["acc"]= lambda p, t : ((p >= th).astype(bool) == t.astype(bool)).mean()
    metric_functions["kl"]= kl#lambda p, t : np.mean(t*np.log((t+1e-10)/(p+1e-10) )+ (1-t)*np.log(((1-t)+1e-10)/((1-p)+1e-10)))

    metrics =[metric_functions[el](pred, target) for el in kwargs["knn_metrics"]]

    #acc = ((pred >= th).astype(bool) == target.astype(bool)).mean()
    #pred = np.array([1-pred,pred]).transpose()
    #f1 = f1_score(target, pred)
    #aur = au_roc(pred, target)
    return metrics, th#(aur, acc, kl), th#f1_score(np.array(classes), pred>0.5)

def kl(pred, target):
    pred = (pred + 1e-7) *(1-2e-7)

    p = np.where(target != 0, target * np.log(target / pred), 0)
    n = np.where((1-target) != 0, (1-target) * np.log((1-target) / (1-pred)), 0)
    #n=0
    return np.mean(p+n)

def print_res_to_latex(res, layers, layer_size, caption="", metrics=None):
    size = len(res[0][0])
    out = ""
    metrics = metrics or ["" for el in range(size)]
    for i in range(size):
        out += "\\begin{figure}\n "
        out += "\\begin{tabular}{|"
        for _ in range(len(layer_size)+1):
            out += "c|"
        out += "}\\hline\n"
        for s in layer_size:
            out += " & " + str(s)
        out += "\\\\ \hline \n"
        for j, l in enumerate(res):
            out += str(layers[j])
            for elm in l:
                el = elm[i]
                out += "& $" + str("{:0.4}".format(float(el))) + " $ "
            out+= "\\\\ \\hline\n"
        out += "\\end{tabular}\n"
        out += "\\caption{" + caption + " " + metrics[i] + "}\n"
        out += "\\end{figure}\n\n "

    return out