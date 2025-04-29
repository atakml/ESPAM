
import networkx as nx
from collections import defaultdict
import numpy as np


from sklearn.linear_model import Lasso, ElasticNet

def re_label(elements, keymap):
    retset = set()
    for k, el in elements.items():
        retset.add(el)
    index = len(keymap)
    keymap.update({el : i+index for i, el in enumerate([el for el in retset if el not in keymap.keys()])})
    retmap = {k : keymap[el] for k, el in elements.items()}
    return retmap, keymap

def step(graph, elements, keymap):
    """
    :param graph: nx graph
    :param elements: mapping i->label
    :param index: next label
    :return:
    """
    els = {el: tuple([elements[el]]+sorted([elements[n] for n in graph.adj[el]])) for el in graph.nodes}
    new_map, index =re_label(els, keymap)
    return new_map, index


def wl(graph, keymap, deep):
    """takes an nx graph and returns a multiset of labels"""
    multiset = defaultdict(int)
    elements={el:(np.argmax(graph.nodes[el]["x"]),) for el in graph.nodes}
    labels, keymap = re_label(elements, keymap)
    for k in labels.values():
        multiset[k]+=1
    for i in range(deep):
        labels, keymap = step(graph, labels,  keymap)
        for k in labels.values():
            multiset[k]+=1

    return multiset,keymap

def featurize(vector, size):
    ret = np.zeros(size)
    for k,v in vector.items():
        if k<size:
            ret[k]=v
    return ret

def wl_kernel(trainset, train_labels, label_map):
    """
    retuns a vector of coeficients
    :param trainset:
    :param label_map:
    :return:
    """
    maxindex = max(label_map.values())+1

    vectorized = [featurize(el, maxindex) for el in trainset]

    lasso = ElasticNet(alpha=0.01,l1_ratio=0.1)
    lasso.fit(vectorized,train_labels)
    return lasso

def test_kernel(k, dset, labels):
    maxindex = len(k.coef_)
    vectorized = [featurize(el, maxindex) for el in dset]
    print(k.score(vectorized, labels))
    print(((k.predict(vectorized)>0.5)==labels).mean())



def kernel(values, g1, g2):
    v1 = featurize(g1, len(values))
    v2 = featurize(g2, len(values))

