import random
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from torch_geometric.data import DataLoader as Dl
from tqdm import tqdm

from model import GCNet


class SaveGrad:
    def __init__(self):
        self.data = list()

    def __call__(self, grad):
        self.data.append(grad)


def extract_data(model: GCNet, dataset, device=torch.device('cpu'), **kwargs):
    table = list()
    model.to(device)
    model.eval()
    newdl = Dl(dataset.dataset, batch_size=1, shuffle=False)
    data_list = list()
    for (k, data), d in tqdm(list(zip(enumerate(newdl), dataset.dataset))):
        data.to(device)
        data_list.append(d)
        register = SaveGrad()
        pred = model.internal_forward(data, register=register)
        scores = pred[-1][0]
        score_max_index = scores.argmax()
        score_max = scores[score_max_index]
        score_max.backward(retain_graph=True)
        for i in range(len(model.layers)):
            table += [((el, gr), k, i) for el, gr in zip(pred[i][0], register.data[i])]
    return table, data_list

def compute_ae_embedding(model, dataset, device, **kwargs):
    model.to(device)
    model.eval()
    table = list()
    data_list = list()
    for (k, data), d in tqdm(list(zip(enumerate(dataset), dataset.dataset))):
        data.to(device)
        data_list.append(d)
        emb = model.encoder(data)
        table +=[(el, k) for el in emb]
    return table, data_list

def molecular_maping(dataset):
    """
    maps each edge to its molecule
    :param dataset:
    :return:
    """
    ret = defaultdict(list)
    for el in dataset:
        ret[el[1]].append(el)
    return ret


def explain(elements, dataset, distance, datalist, cl, **kwargs):
    m = torch.tensor([2.0 ** 1000])
    ret = -1
    for el, _ in elements:
        for mol in dataset.values():
            for v, key in mol:
                if datalist[key].y == cl:
                    d = distance(el, v)
                    if d < m:
                        m = d
                        ret = key
    return ret


from dataloader import atoms


def select_nodes(vect, element, k):
    els = list()
    for i in range(k):
        ok = True
        el = None
        while ok:
            el = random.randint(0, len(vect) - 1)
            if el not in els and atoms[element.x[el].argmax()] != "H":
                ok = False
        els.append(el)
    return els


def select_nodes2(vect, element, k):
    """
    prend les k meilleurs éléments selon le gradient
    :param vect: 
    :param element: 
    :param k: 
    :return: 
    """
    els = [(sum(el[0][1]), i) for i, el in enumerate(vect) if atoms[element.x[i].argmax()] != "H"]
    els.sort(key=lambda x: x[0], reverse=True)
    return [el[1] for el in els[:k]]


def rdm_expl(elements, dataset, distance, datalist, cl, k, **kwargs):
    """
    plus très random comme explication
    TODO: passer sur un algo
    :param elements:
    :param dataset:
    :param distance:
    :param datalist:
    :param cl:
    :param k:
    :param kwargs:
    :return:
    """
    m = torch.tensor([2.0 ** 1000])
    ret = -1
    ret_atoms, ret_mol = None, None
    for key, mol in dataset.items():
        # if torch.abs(datalist[key].y - cl)<epsilon:

        if datalist[key].y == cl:
            d = torch.tensor(0.)
            expl_att = list()
            for (el,v), _ in elements:
                mi = torch.tensor([2.0 ** 1000])
                at=None
                for i, ((v, _), _) in enumerate(mol):
                    dis = distance(el, v)
                    if dis < mi and i not in expl_att:
                        mi = dis
                        at = i
                d += mi
                expl_att.append(at)
            if d < m:
                m = d
                ret = key
                ret_atoms = expl_att
                ret_mol = mol
    return ret, ret_atoms, ret_mol


import math


def clusters(data, data_list, **kwargs):
    vectors = ([el[0].tolist() for el in data])
    scores = list()
    max_cluster = 50
    for k in range(2, max_cluster):
        kmean = KMeans(n_clusters=k, n_jobs=20, n_init=20)
        kmean.fit(vectors)
        classes = kmean.predict(vectors)
        scores.append(kmean.score(vectors))
        entropy = [[] for _ in range(k)]
        for i, el in enumerate(classes):
            entropy[el].append(data_list[data[i][1]].y)
        entropy = [(torch.tensor(el).float().mean()) for el in entropy]
        var = (torch.tensor(entropy).pow(2).mean() - torch.tensor(entropy).mean().pow(2)).float()
        print(var)
        # torch.tensor(entropy).tolist()
        entropy = [(len(el), torch.tensor(el).to(float).mean()) for el in entropy]

        # entropy = [-(el*math.log(el,2)+ (1-el)*math.log(1-el,2)) for k, el in entropy]
        # entropy = [-(k+0.)/len(vectors)*(el*math.log(el,2)+ (1-el)*math.log(1-el,2)) for k, el in entropy]

        print(torch.tensor(entropy).sum())

    l = len(vectors)
    scores = [math.log(-el) for el in scores]
    plt.plot(range(2, max_cluster), scores, '-')
    plt.show()


def pareto_expl(elements, dataset, igrads, distance, datalist, cl, k, **kwargs):
    def dominate(a, b):
        if a[0] < b[0] and a[1] >= b[1]:
            return True
        if a[0] == b[0] and a[1] > b[1]:
            return True
        return False

    expls = list()
    for key, el in dataset.items():
        if datalist[key].y == cl:
            exp = pick_explanations(el, k, 20)
            expl = score_explanation_igrad(elements, exp, igrads[key], distance)
            expls += expl

    par = extract_pareto(expls, dominate)
    return random.choice(par)


def score_explanation_igrad(elements, expls, igrad, distance):
    ret = list()
    for el, indexes in expls:
        d = 0
        elc = el.copy()
        for (elm, _), _ in elements:
            dmin = 10.0 ** 20
            imin = None
            for i, ((x, _), _) in enumerate(elc):
                tmp = (elm - x).norm(2).to(float)
                if tmp < dmin:
                    dmin = tmp
                    imin = i
            d += dmin
            del elc[imin]
        q = 0
        for i in indexes:
            q += igrad[0][i].norm(2).to(float)
        ret.append(((d, q), el, indexes))
    return ret


def score_explanation(elements, expls, distance):
    ret = list()
    for el, indexes in expls:
        d = 0
        elc = el.copy()
        for (elm, _), _ in elements:
            dmin = 10.0 ** 20
            imin = None
            for i, ((x, _), _) in enumerate(elc):
                tmp = distance(elm, x).to(float)
                if tmp < dmin:
                    dmin = tmp
                    imin = i
            d += dmin
            del elc[imin]
        q = 0
        for (_, v), _ in el:
            q += v.norm(2).to(float)
        ret.append(((d, q), el, indexes))
    return ret


def pick_explanations(el, k, n):
    ret = list()
    for i in range(n):
        if len(el) >= k:
            x = random.sample(range(len(el)), k)
            ret.append([[el[k] for k in x], x])
    return ret


def extract_pareto(expls: list, dominate):
    par = list(map(lambda x: [x, 1], expls))
    for i in range(len(par)):
        if par[i][1]:
            for j in range(len(par)):
                if par[j][1]:
                    if dominate(par[j][0][0], par[i][0][0]):
                        par[i][1] = 0
                        break
                    if dominate(par[i][0][0], par[j][0][0]):
                        par[j][1] = 0
    pareto = filter(lambda x: x[1] == 1, par)
    pareto = list(map(lambda x: x[0], pareto))
    return pareto
