
import pysubgroup as ps
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils.tb import plot_mol_emp
from torch_geometric.data import DataLoader as Dl, Data, Batch
from torch_geometric.utils.convert import to_networkx
import os
from torch.utils.tensorboard import SummaryWriter
from utils.LTSbridge import read_subgroup, run_LTS

def tb_rules(dataset, dataframe, rule, layer, remap_labels= None, **kwargs):
    writer = SummaryWriter("runs/"+ kwargs["name"]+"l_"+str(layer)+str(rule))
    dset=[el for el in Dl(kwargs["trainset"].dataset.dataset, batch_size=1)]

    rl=remap_labels["0"].tolist()

    if remap_labels is not None:
        dset=[dset[el] for  el in rl]
        """out= [0 for _ in dset]
        for i,el in enumerate(rl):
            out[el] = dset[i]
        dset=out"""

    #id2 = d2dataframe.loc[-(dataframe["id"].isin(dataframe[rule.covers(DD)]["id"]))]
    ids= dataframe[rule.covers(dataframe)]
    for i, el in enumerate(tqdm(dset)):

        index = ids.index[ids["id"] == i].tolist()
        i0 = dataframe.index[dataframe["id"] == i][0]
        index = [i-i0 for i in index]

        if len(index):
            atoms = get_neigborhood(el, index, layer)
            fig = plot_mol_emp(el, atoms, index, **kwargs)
            writer.add_figure("img/"+str(i), fig)
    writer.close()


def plot_dset(**kwargs):
    writer = SummaryWriter("runs/"+ kwargs["name"]+"_dset")
    dset=Dl(kwargs["trainset"].dataset, batch_size=1)
    for i, el in enumerate(tqdm(dset)):
        fig = plot_mol_emp(el, [], [], **kwargs)

        writer.add_figure("img/"+str(i)+"cls_"+str(el.y.numpy()[0]), fig)
    writer.close()

def reorder(graph, dset, cls, elements_add):

    positives= list()
    negatives= list()
    for el in molmap:
        if dset[el].y.numpy()[0]==1:
            positives.append(el)
        else :
            negatives.append(el)
    for i in range(elements_add):
        el = np.randint(0, len(dset))
        while el in cls:
            el = np.randint(0, len(dset))
        negatives.append(el)
    return positives+negatives+ len(positives)

def get_featured_subgraphs_no_label(ids, dframe, dset, radius, cls, **kwargs):
    graphs = list()
    molmap = list()
    for i, moln in enumerate(tqdm(ids)):
        if cls is None or dset[moln].y.numpy()[0]== cls:
            idx = ids.index[i]
            i0 = dframe.index[dframe["id"] == moln][0]
            index = [idx-i0]
            mol = to_networkx(dset[moln],node_attrs="x").to_undirected()
            atoms = get_neigborhood(dset[moln], index, radius)
            ego_network = nx.Graph(mol.subgraph(atoms))
            for n in ego_network.nodes():
                ego_network.nodes[n]["x"]= [1]
            graphs.append(ego_network)
            molmap.append(moln)

    return graphs, molmap

import random

def get_freatured_subgraphs(ids, dframe, dset, radius, cls, **kwargs):
    graphs = list()
    molmap = list()
    graphsneg = list()
    molmapneg = list()
    distribution = list()
    values = [[1,0], [0,1]]

    for i, moln in enumerate(tqdm(ids)):
        if cls is None or dset[moln].y.numpy()[0]== cls:
            idx = ids.index[i]
            i0 = dframe.index[dframe["id"] == moln][0]
            index = [idx-i0]
            mol = to_networkx(dset[moln],node_attrs="x").to_undirected()
            atoms = get_neigborhood(dset[moln], index, radius)
            activated_index = [ids.index[i] for i, el in enumerate((ids == moln ).tolist()) if el]
            ego_network = nx.Graph(mol.subgraph(atoms))
            for n in ego_network.nodes():
                ego_network.nodes[n]["x"]= values[int(n+i0 in activated_index)]

            graphs.append(ego_network)
            molmap.append(moln)
            distribution.append(len(activated_index))


            random.shuffle(atoms)
            for a in atoms:
                if a not in activated_index:
                    atoms =get_neigborhood(dset[moln], index, radius)
                    ego_network2 = nx.Graph(mol.subgraph(atoms))
                    activatepd_index = [ids.index[i] for i, el in enumerate((ids == moln).tolist()) if el]
                    for n in ego_network.nodes():
                        ego_network2.nodes[n]["x"] = values[int(n + i0 in activatepd_index)]
                    graphsneg.append(ego_network2)
                    molmapneg.append(moln)

    return graphs, molmap, distribution, graphsneg,molmapneg

import random


def get_un_freatured_subgraphs(ids, dset, pos_sample, radius, nitems):
    graphs = list()
    molmap = list()
    values = [[1,0], [0,1]]
    for i in range(nitems):
        el = np.random.randint(0, len(dset))
        while el in ids.tolist():
            el = np.random.randint(0, len(dset))
        mol = to_networkx(dset[el], node_attrs="x").to_undirected()
        #graphs.append(mol)
        atomn = np.random.randint(0, len(mol.nodes))

        atoms = get_neigborhood(dset[el], [atomn], radius)

        ego_network = nx.Graph(mol.subgraph(atoms))

        n_act = np.random.randint(0, len(pos_sample))
        activated_nodes = list(ego_network.nodes())
        random.shuffle(activated_nodes)
        for node in ego_network.nodes():
            ego_network.nodes[node]["x"]= values[int(node in activated_nodes[:n_act])]

        graphs.append(ego_network)

        molmap.append(el)
    return graphs, molmap

def get_un_featured_subgraphs_no_labels(ids, dset, radius, nitems):
    graphs = list()
    molmap = list()
    for i in range(nitems):
        el = np.random.randint(0, len(dset))
        while el in ids.tolist():
            el = np.random.randint(0, len(dset))
        mol = to_networkx(dset[el], node_attrs="x").to_undirected()
        #graphs.append(mol)
        atomn = np.random.randint(0, len(mol.nodes))

        atoms = get_neigborhood(dset[el], [atomn], radius)

        ego_network = nx.Graph(mol.subgraph(atoms))
        for n in ego_network.nodes():
            ego_network.nodes[n]["x"]= [1]
        graphs.append(ego_network)

        molmap.append(el)
    return graphs, molmap

def iso_map(graphs):
    iso_map = [1 for _ in graphs]
    node_match = lambda a,b: a["x"] == b["x"]
    edge_match= lambda a,b:  a == b
    for i, g1 in enumerate(tqdm(graphs)):
        if iso_map[i]:
            for j, g2 in enumerate(graphs):
                if i!=j and nx.is_isomorphic(g1,g2, node_match=node_match, edge_match=edge_match):
                    #iso_map[i] = 0
                    iso_map[j] = 0
    return iso_map

def get_best_subgraph(dframe, rule, radius, **kwargs):

    eps = 10e-10

    ids = dframe[rule.covers(dframe)]
    ids = pd.to_numeric(ids["id"], downcast="integer")
    dset = [el for el in Dl(kwargs["trainset"].dataset, batch_size=1)]
    #tb_rules(None, dframe, rule, radius, **kwargs)


    k = 70
    ids = ids.sample(frac=1)
    nPositive = len(dframe[rule.covers(dframe)].groupby("id"))
    nNegative = nPositive


    positives, mapPos = get_featured_subgraphs_no_label(ids, dframe, dset, radius, None)
    salt, mapSalt = get_un_featured_subgraphs_no_labels(ids, dset, radius, nNegative)

    #positives, mapPos, distribution, negs, mapNegs = get_freatured_subgraphs(ids, dframe, dset, radius, None)
    #salt, mapSalt = get_un_freatured_subgraphs(ids, dset, distribution, radius, nNegative)
    #negs = negs[:k]
    #mapNegs = mapNegs[:k]

    positives = positives[:k]
    mapPos = mapPos[:k]

    salt = salt[:k]
    mapSalt = mapSalt[:k]

    run_LTS(positives + salt, len(positives),**kwargs)
    molmap = mapPos + mapSalt#mapSalt[:100]

    #run_LTS(positives +negs, len(positives),**kwargs)
    #molmap = mapPos + mapNegs#mapSalt[:100]

    dirname = "datasets/subgraphs"

    retgrs = read_subgroup(os.path.join(dirname, "output_LTSDFS"), **kwargs)


    unique_elements = [i for i, el in enumerate(iso_map(retgrs[2])) if el]
    #unique retgrs
    grs = [[retgrs[i][el] for el in unique_elements] for i in range(3)]
    gr = [to_graphs_indices(g, molmap) for g in grs[:2]]
    #gr = [gr[el] for el in unique_elements]
    best_rule = [np.log((len(p)/nPositive)/(eps+len(n)/nNegative)) for n, p in zip(*gr)]
    best_index = np.argmax(best_rule)
    texts = ["pos"+str(len(p))+":"+str(len(positives))+ "neg_"+str(len(n))+":"+str(k)+ " score : "+str(best_rule[i]) for i,(n, p) in enumerate(zip(*gr))]
    #best_rule = np.argmin([ np.log(len(p)/(len(n)+eps)) for n, p in zip(*gr)])
    get_prr(gr[1][best_index],gr[0][best_index], grs[2][best_index], **kwargs)
    print(best_rule)
    plot_graph_list(grs[2], texts, rule,**kwargs)
    #plot_base_mol(grs, dset, rule, molmap, nPositive, nNegative, "positive", **kwargs)
    plot_support(gr[0], dset, rule, "negative", **kwargs)
    plot_support(gr[1], dset, rule, "positive", **kwargs)
    #best_rule = [ np.log(len(p)/(len(n)+eps)) for n, p in zip(*gr)]
    #best_rule = np.argmin([ np.log(len(p)/(len(n)+eps)) for n, p in zip(*gr)])
    return best_rule, gr

def get_prr(pos, neg, subgraph, **kwargs):
    dset = [el for el in Dl(kwargs["trainset"].dataset, batch_size=1)]

    selected_good = [0 for _ in dset]
    #pertinent = 0
    pertinent_edges = 0
    for gr, nodes in pos:
        mol = dset[gr]
        mol = to_networkx(mol,node_attrs="z").to_undirected()
        pertinent = {(a, b) for a,b in mol.edges() if (mol.nodes[a]["z"]>0 and mol.nodes[b]["z"]>0)}

        #pertinent += sum([mol.nodes[el]["z"]>0 for el in mol.nodes])
        #intersect = mol.subgraph(nodes)
        intersect = {(nodes[a], nodes[b]) for  a,b in subgraph.edges()}

        #selected_nodes = sum([intersect.nodes[el]["z"]>0 for el in intersect.nodes])
        pertinent_edges += len(pertinent)
        selected_good[gr] = max(selected_good[gr], len(pertinent.intersection(intersect)))
        #correct_edges =
        #selected_edges

    precision = sum(selected_good)/((len(pos)+len(neg))* len(subgraph.edges()))

    rappel = sum(selected_good)/(pertinent_edges)
    print(precision, rappel)
    return precision, rappel


def plot_graph_list(subgraphs, texts, rule, **kwargs):

    writer = SummaryWriter("runs/" + kwargs["name"] + "_subgraphs2_bfs_rule_" + str(rule))
    for i, g in enumerate(subgraphs):
        fig = plt.figure(figsize=(4, 4))

        nx.draw_kamada_kawai(g, with_labels=False)
        #nx.draw_kamada_kawai(g)

        labels= {node: g.nodes[node]["x"]#"%1.2f"%float(mol_values[i][0][1].sum()))
             for i,node in enumerate(g.nodes())}
        nodes = {node for node in g.nodes() }
        nx.draw_kamada_kawai(g, node_size=1, with_labels=False)

        nx.draw_kamada_kawai(g, nodelist=nodes, labels=labels, with_labels=True)
        writer.add_figure("img/" + str(i)+"_"+ texts[i], fig)
    writer.close()

def to_graphs_indices(subgraphs, molmap):
    out_gr= list()
    for i,grs in enumerate(subgraphs):
        mm = set()
        gr = list()
        for a, b in grs:
            id = molmap[a]
            if id not in mm:
                mm.add(id)
                gr.append((id, b))
        out_gr.append(gr)
    return out_gr

def plot_base_mol(graph, dset, rule,molnum, pos, neg, filename, **kwargs):
    pltgrs = list()
    for i, g in enumerate(graph[1]):
        moln = molnum[i]
        text = "n_"+ str(i) + "mol_N_" + str(moln) +\
              "_pos_" + str(len(graph[1][i])) + "-" + str(pos) +\
              "_neg_" + str(len(graph[0][i])) + "-" + str(neg) +\
              "_class_" + str(dset[moln].y.numpy()[0])
        pltgrs.append((next((a,b) for a, b in g if a==moln), text))
        #str = len(gr) +":" len(pos)+"neg" n  neg[i]

    plot_subgraphs(pltgrs, dset, rule, "sgs"+filename, **kwargs)

def plot_support(graph, dset, rule,filename, **kwargs):
    for i, g in enumerate(graph):
        g = [(gr, "n_"+str(i)+ "_graphN_"+str(gr[0]) +"_class_"+ str(dset[gr[0]].y.numpy()[0])) for gr in g]
        plot_subgraphs(g, dset, rule,"support_"+filename, **kwargs)

def plot_subgraphs(grs, dset, rule, plot_name, **kwargs):
    writer = SummaryWriter("runs/"+ kwargs["name"]+"_"+plot_name+"_bfs_rule_"+str(rule))
    for i, ((el, a), name) in enumerate(grs) :
        fig = plot_mol_emp(dset[el], a, [], **kwargs)
        writer.add_figure("img/"+ name, fig)
    writer.close()


def to_dataframe(data, **kwargs):
    dset = kwargs["trainset"].dataset
    elements=  [dset.dataset[el] for el in dset.indices]
    atoms = [x.numpy() for el in elements for x in el.x]

    layersize = kwargs["hidden_dims"][0]
    nlayer = len(kwargs["hidden_dims"])
    d2 = np.array([[data[0][el][1]]+ [x  for l in range(nlayer) for x in data[l][el][0][0].tolist()] for el in range(len(data[0])) ])
    dd = pd.DataFrame(d2, columns=["id"]+["l_"+ str(i//layersize) +"c_"+ str(i%layersize) for i in range(nlayer*len(data[0][0][0][0]))])
    dd = pd.concat([dd,pd.DataFrame(atoms, columns=["a_"+str(i) for i in range(len(atoms[0]))])], axis=1)
    labels = pd.DataFrame({"class": [kwargs["trainset"].dataset[el[1]].y.numpy()[0] for el in data[0] ]})
    dd = pd.concat([dd, labels], axis=1)
    dd.update(dd.filter(regex="l_[0-9]*c_[0-9]*")>0)
    return dd

def relabel_model_output(dframe, **kwargs):
    labels = pd.DataFrame({"class": [kwargs["model_output"][int(el)] for el in dframe["id"] ]})
    dframe["true_class"] = dframe["class"]
    dframe["class"] = labels["class"]
    return dframe

def to_dataframe_mean(data, **kwargs):
    dd= to_dataframe(data, **kwargs)

    return dd.groupby("id").max()



def get_dis_rules(data, layer, **kwargs):
    dd = to_dataframe(data, **kwargs)
    d2 = dd.copy()
    rules = list()
    bt = [MyBinaryTarget('class', 0), MyBinaryTarget('class', 1)]

    r = [None, None]
    for i in range(10):
        rule = get_rules(d2, (i+1)%2)
        print(bt[i % 2].get_base_statistics(rule, d2))

        d2 = d2.loc[-(dd["id"].isin(dd[rule.covers(dd)]["id"]))]
        rules.append(rule)
        if r[i%2]:
            r[i % 2] = ps.Disjunction([r[i % 2], ps.Conjunction(ps.Disjunction([r[i % 2] ,rule]))])
        else :
            r[i % 2] = rule

        if r[i % 2]:
            print(bt[i % 2].get_base_statistics(r[i % 2], dd))
    return r

def frequent_motifs(data,layer, **kwargs):
    d2 = np.array([[data[0][el][1]]+ [x > 0 for l in range(layer) for x in data[l][el][0][0].tolist()] for el in range(len(data[0])) ])
    dd = pd.DataFrame(d2, columns=["id"]+[str(i) for i in range(layer*len(data[0][0][0][0]))])
    labels = pd.DataFrame({"class": [kwargs["trainset"].dataset[el[1]].y.numpy()[0] for el in data[layer-1] ]})

    target = ps.FITarget()
    searchspace = ps.create_selectors(dd, ignore=["id","class"])
    task = ps.SubgroupDiscoveryTask (
        dd,
        target,
        searchspace,
        result_set_size=10,
        depth=2,
        qf=ps.CountQF())##ps.StandardQF(a=0.2))
    result = ps.SimpleDFS().execute(task)
    print(result.to_dataframe())

def get_sum_rules(data, **kwargs):
    dd = to_dataframe_mean(data, **kwargs)
    dd["class"] = dd["class"]>0
    target = MyBinaryTarget('class', 1)
    searchspace = ps.create_selectors(dd, ignore=['class'])
    task = ps.SubgroupDiscoveryTask (
        dd,
        target,
        searchspace,
        result_set_size=10,
        depth=5,
        qf=ps.StandardQF(a=0.3))
    result = ps.BeamSearch().execute(task)
    return result
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from matplotlib.pyplot import figure

def get_decision_tree(data, **kwargs):
    dframe = to_dataframe(data, **kwargs)
    rules = rules_layers(dframe, **kwargs)
    rules_flat = [r for rl in rules for r in rl]
    cv=list()
    for rule in rules_flat:
        cv.append(rule.covers(dframe).tolist())
    cc= np.array(cv).transpose()
    dd2 = pd.DataFrame(cc, columns=["r_"+str(i)for i in range(len(rules_flat))])
    dd2= pd.concat([dframe["id"], dd2,dframe["class"]],axis=1)
    dd = dd2.groupby("id").max()
    #DD = to_dataframe_mean(data, **kwargs)

    dt=DecisionTreeClassifier(max_depth=6)
    dt.fit(dd.filter(regex="r_[0-9]*"),dd["class"])

    #_, xx = plt.subplots(1,2, figsize=(6,3))
    figure(num=None, figsize=(80,20))

    plot_tree(dt, filled=True)
    #plot_tree(dt, feature_names=["r_"+str(i)for i in range(len(rules_flat))], filled=True)

    #figure(num=None, figsize=(20,20))

    plt.show()
    return dt
from functools import reduce
def plot_rules(dset, **kwargs):
    rule_list= {
        20:[("l_4","c_9",1),("l_4","c_4",1),("l_4","c_6",1),("l_4","c_8",1)],
        4:[("l_0","c_7",1),("l_0","c_4",1)],
        13:[("l_3","c_2",1),("l_3","c_4",1),("l_3","c_7",1),("l_3","c_5",1),("l_3","c_8",1),("l_3","c_0",1)]
    }
    from experiments import csv_to_df

    """dframe = to_dataframe(dset,**kwargs)
    if "model_labels": #TODO
        dframe = relabel_model_output(dframe, **kwargs)"""
    dframe, index = csv_to_df()
    kwargs["name"]+="_tt"

    for k, v in rule_list.items():
        rule = ps.Conjunction([ps.EqualitySelector(el[0]+el[1], 1)for el in v])
        #ids= dframe[rule.covers(dframe)].group_by("id")
        #ids = pd.to_numeric(ids["id"], downcast="integer")
        #dset = [index.iloc[el][1] for el in Dl(kwargs["trainset"].dataset, batch_size=1)]

        tb_rules(None, dframe, rule, 4,remap_labels=index,  **kwargs)
def workflow(dset, **kwargs):
    #plot_dset(**kwargs)
    dframe = to_dataframe(dset,**kwargs)
    if "model_labels": #TODO
        dframe = relabel_model_output(dframe, **kwargs)

    rules = rules_layers(dframe, **kwargs)
    rules_flat = [r for rl in rules for r in rl]
    """
    layer = 3
    l = "".join([str(i) for i in range(layer)])
    DD = pd.concat([pd.to_numeric(dframe["id"], downcast="integer"), dframe["class"], dframe.filter(regex="l_[" + l + "]*c_[0-9]*")],
                   axis=1)

    # DD["id"].to_numeric(downcast="integer")

    target = MyBinaryTarget('class', 1)
    searchspace = ps.create_selectors(DD, ignore=['class', "id"])
    task = ps.SubgroupDiscoveryTask(
        DD,
        target,
        searchspace,
        result_set_size=10,
        depth=3,
        qf=MyQualityFunction(a=0.8))  # ps.StandardQF(a=0.2))
    result = ps.BeamSearch().execute(task)"""
    best_rule = max(rules_flat, key=lambda x :x[2]/(x[1]+10))
    #best_rule = min(rules_flat, key=lambda x : (x[2]<10)*100+x[2] )

    """print("rule : ",result.to_dataframe()["subgroup"][0],
          " size sg :", result.to_dataframe()["size_sg"][0],
          " positive :", result.to_dataframe()["positives_sg"][0])"""
    print("rule : ", best_rule[0],
          " size sg :", best_rule[1],
          " positive :", best_rule[1])
    #extract_subgraph(dset, dframe,best_rule[0], 4, **kwargs)

    best = get_best_subgraph(dframe,best_rule[0], best_rule[3], **kwargs)
    #best = get_best_subgraph(dframe,best_rule[0], 2, **kwargs)

    print(best[0])
    #extract_subgraph(dset, )


def getrules(data, layer, **kwargs):
    dd = to_dataframe(data,**kwargs)
    #DD = pd.concat([pd.to_numeric(d2["id"],downcast="integer"),d2["class"], d2.filter(regex="l_"+str(layer)+"*c_[0-9]*") ], axis=1)
    l= "".join([str(i)for i in range(layer)])
    dd = pd.concat([pd.to_numeric(dd["id"],downcast="integer"),dd["class"], dd.filter(regex="l_["+l+"]*c_[0-9]") ], axis=1)

    #DD["id"].to_numeric(downcast="integer")

    target = MyBinaryTarget('class', 1)
    searchspace = ps.create_selectors(dd, ignore=['class',"id"])
    task = ps.SubgroupDiscoveryTask(
        dd,
        target,
        searchspace,
        result_set_size=10,
        depth=2,
        qf=MyQualityFunction(a=0.2))#ps.StandardQF(a=0.2))
    result = ps.BeamSearch().execute(task)
    tb_rules(kwargs["trainset"], dd, result.results[0][1], layer, **kwargs)

    print(result.to_dataframe())

def get_rule_set(dframe, regex, result_size=50, factor=0.5, layer=0):
    dd = pd.concat(
        [pd.to_numeric(dframe["id"], downcast="integer"), dframe["class"], dframe.filter(regex=regex)],
        axis=1)
    # DD["id"].to_numeric(downcast="integer")

    target = MyBinaryTarget('class', 1)
    searchspace = ps.create_selectors(dd, ignore=['class', "id"])
    task = ps.SubgroupDiscoveryTask(
        dd,
        target,
        searchspace,
        result_set_size=result_size,
        depth=2,
        qf= MyQualityFunction(a=factor))  #
    result = ps.BeamSearch(beam_width=500).execute(task)
    #tb_rules(kwargs["trainset"], DD, result.results[0][1], layer, **kwargs)
    return [(r[1],r[2][0],r[2][1],layer) for r in result.results]

def rules_layers(dframe, **kwargs):
    new_rules = list()
    for f in [0.1, 0.2, 0.5, 0.8, 1.2]:
        new_rules += get_rule_set(dframe, "a_[0-9]*", result_size=20, factor=f,layer=0)
    new_rules = eliminate_same_rules(dframe, new_rules)

    rules = [new_rules]
    for i in range(len(kwargs["hidden_dims"])):
        new_rules = list()
        for f in [0.1, 0.2, 0.5, 0.8, 1.2]:
            new_rules += get_rule_set(dframe, "l_" + str(i) + "*c_[0-9]*", result_size=20, factor=f,layer=i+1)
        new_rules = eliminate_same_rules(dframe, new_rules)
        rules.append(new_rules)
    return rules

def gen_rules(data, **kwargs):
    dframe = to_dataframe(data, **kwargs)
    rules = rules_layers(dframe, **kwargs)

    mat = rule_matrix(rules, dframe, **kwargs)
    return mat



def eliminate_same_rules(dframe, rules,coverage=2):
    remove = list()
    covers = np.array([rule[0].covers(dframe).tolist() for rule in rules])
    for i in range(len(rules)):
        if i not in remove:
            for j in range(i+1, len(rules)):
                if j not in remove:
                    if (covers[i]!=covers[j]).sum()/max(2,covers[i].sum(),covers[j].sum())<coverage/100.:
                        remove.append(j)
    return [rules[i] for i in range(len(rules)) if i not in remove]

def rule_matrix(rules, data, **kwargs):
    rules_flat = [r for rl in rules for r in rl]
    confusion_matrix= np.zeros((len(rules_flat), len(rules_flat)))
    cv=list()
    for rule in rules_flat:
        cv.append(rule.covers(data).tolist())
    dd2 = pd.concat([pd.DataFrame(cv).transpose()])

    for i in range(len(rules_flat)):
        for j in range(len(rules_flat)):
            confusion_matrix[i,j]= (dd2[i]*dd2[j]).sum()
    #d2 = d2.loc[-(DD["id"].isin(DD[rule.covers(DD)]["id"]))]

    #for i in range(len(rules_flat)):
    return confusion_matrix

def get_rules(data, target):
    target = MyBinaryTarget('class', target)
    searchspace = ps.create_selectors(data, ignore=['class',"id"])
    task = ps.SubgroupDiscoveryTask (
        data,
        target,
        searchspace,
        result_set_size=10,
        depth=3,
        qf=MyQualityFunction(0.2))#ps.StandardQF(a=5))
    result = ps.BeamSearch().execute(task)
    return result.results[0][1]

def get_rules_set_help(data, target):
    target = MyBinaryTarget('class', target)
    searchspace = ps.create_selectors(data, ignore=['class',"id"])
    task = ps.SubgroupDiscoveryTask (
        data,
        target,
        searchspace,
        result_set_size=10,
        depth=3,
        qf=MyQualityFunction(0.2))#ps.StandardQF(a=5))
    result = ps.BeamSearch().execute(task)
    return result.results[0][1]

import networkx as nx

def get_neigborhood(molecule, nodes, radius):
    mol = to_networkx(molecule, node_attrs="x").to_undirected()
    nn = nodes.copy()
    for i in range(radius):
        nn += [el for n in nn for el in list(mol.adj[n]) ]
        nn = list(set(nn))
    #nbh = nx.Graph(mol.subgraph(list(g1.adj[nodes])))
    return nn





def get_matrices(data):
    i=0
    while i< len(data)-1:
        vlist = list()

        for j in range(30):
            #while i<len(data)-1 and data[i][1] == data[i+1][1]:
            if  i<len(data)-1 and data[i][1] == data[i+1][1]:
                vlist.append(tuple([int(el >0) for el in data[i][0][0].tolist()]))
                i+=1
            else :
                vlist.append(tuple([0 for _ in vlist[-1]]))

        yield np.array(tuple(vlist)[0])
        i+=1

"""
class MySelector:
    def __init__(self, layer, subst):
        self.layer = layer
        self.substr = substr

    def covers(self, df):
        for el in df[layer]:
            el
        return df[self.layer].str.contains(self.substr).to_numpy()

        return df[self.column].str.contains(self.substr).to_numpy()


contains_selector = StrContainsSelector('Sur_name','m')
print(contains_selector.covers(df))
"""
from collections import namedtuple
class MyQualityFunction(ps.SimplePositivesQF, ps.BoundedInterestingnessMeasure):
    @staticmethod
    def standard_qf(a, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup):
        if not hasattr(instances_subgroup, '__array_interface__') and (instances_subgroup == 0):
            return np.nan
        p_subgroup = np.divide(positives_subgroup, instances_subgroup)
        #if instances_subgroup == 0:
        #    return 0
        #p_subgroup = positives_subgroup / instances_subgroup
        p_dataset = positives_dataset / instances_dataset
        return (instances_subgroup / instances_dataset) ** a * (p_subgroup - p_dataset)

    def __init__(self, a=5):
        """
        Parameters
        ----------
        a : float
            exponent to trade-off the relative size with the difference in means
        """
        self.a = a
        super().__init__()

    def calculate_constant_statistics(self, data, target):
        assert isinstance(target, ps.BinaryTarget)
        self.positives_atomes = target.covers(data)#.groupby("id")["class"].max()
        self.positives = data[self.positives_atomes].groupby("id")["class"].max()==target.target_selector.attribute_value
        self.dataset_statistics = ps.SimplePositivesQF.tpl(data["id"].max(), np.sum(self.positives))

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, size_sg = ps.get_cover_array_and_size(subgroup, len(self.positives), data)
        cc= data[cover_arr].groupby("id")["class"].max()
        cc1 = cc.reindex(range(max(self.positives.index)+1))
        return ps.SimplePositivesQF.tpl(len(cc), np.count_nonzero(self.positives[cc1==target.target_selector.attribute_value]))

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return MyQualityFunction.standard_qf(self.a, dataset.size_sg, dataset.positives_count, statistics.size_sg, statistics.positives_count)

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return MyQualityFunction.standard_qf(self.a, dataset.size_sg, dataset.positives_count, statistics.positives_count, statistics.positives_count)

    def optimistic_generalisation(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        pos_remaining = dataset.positives_count - statistics.positives_count
        return MyQualityFunction.standard_qf(self.a, dataset.size_sg, dataset.positives_count, statistics.size_sg + pos_remaining, dataset.positives_count)




class MyBinaryTarget(ps.BinaryTarget):
    def __init__(self, target_attribute=None, target_value=None, target_selector=None):
        super().__init__(target_attribute, target_value, target_selector)

    def get_base_statistics(self, subgroup, data):
        cover_arr, size_sg = ps.get_cover_array_and_size(subgroup, len(data), data)
        cc= data[cover_arr].groupby("id")["class"].max()==self.target_selector.attribute_value

        mols = data.groupby("id").max()
        instances_subgroup = len(cc)
        positives_dataset = (mols["class"]==self.target_selector.attribute_value).sum()
        instances_dataset = len(mols)
        positives_subgroup = sum(cc)
        return instances_dataset, positives_dataset, instances_subgroup, positives_subgroup



