
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx

from dataloader import atoms


from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch_geometric.data import DataLoader as Dl, Data, Batch
import numpy as np

def get_molecule_plot(molecule, **kwargs):
    ret=plt.figure(figsize=(4,4))
    #for i, el in enumerate(molecules):
    mol = to_networkx(molecule,node_attrs="x").to_undirected()
    labels= {node: kwargs["atoms"][el.argmax()]#"%1.2f"%float(mol_values[i][0][1].sum()))
             for i,(node, el) in enumerate(zip(mol.nodes(), molecule.x)) if kwargs["atoms"][el.argmax()]!="H"}
    nodes = {node for node, el in zip(mol.nodes(), molecule.x) if kwargs["atoms"][el.argmax()]!="H"}
    #plt.subplot(size[0],size[1], i+1)
    nx.draw_kamada_kawai(mol, node_size=1, with_labels=False)
    nx.draw_kamada_kawai(mol, nodelist=nodes, labels=labels, with_labels=True)
    return ret

def plot_mol_emp(molecule, nodeslist, centers, **kwargs):
    ret=plt.figure(figsize=(4,4))
    #for i, el in enumerate(molecules):
    mol = to_networkx(molecule,node_attrs="x").to_undirected()
    labels= {node: kwargs["atoms"][el.argmax()]#"%1.2f"%float(mol_values[i][0][1].sum()))
             for i,(node, el) in enumerate(zip(mol.nodes(), molecule.x)) if kwargs["atoms"][el.argmax()]!="H"}
    nodes = {node for node, el in zip(mol.nodes(), molecule.x) if kwargs["atoms"][el.argmax()]!="H"}
    #plt.subplot(size[0],size[1], i+1)
    nx.draw_kamada_kawai(mol, node_size=1, with_labels=False)

    #nx.draw_kamada_kawai(mol, nodelist=nodes, labels=labels, with_labels=True)
    nx.draw_kamada_kawai(mol, nodelist=nodeslist, node_color="r")
    nx.draw_kamada_kawai(mol, nodelist=centers, node_color="y")

    return ret

def write_dset(run,dataset, **kwargs):
    writer = SummaryWriter("runs/"+ kwargs["name"]+" "+dataset)
    dset=Dl(kwargs[dataset].dataset, batch_size=1)
    for i, el in enumerate(tqdm(dset)):
        #figs= [get_molecule_plot(el) for el in tqdm(dset)]
        writer.add_figure("img/"+str(i), get_molecule_plot(el,**kwargs))
    writer.close()
from multiprocessing import Pool
import functools


def nearest(run, dataset, distance_matrix, **kwargs):
    writer = SummaryWriter("runs/distance "+run+" "+kwargs["name"]+"/"+ dataset)

    trainset=list(Dl(kwargs["trainset"].dataset, batch_size=1, shuffle=False))

    dset=list(Dl(kwargs[dataset].dataset, batch_size=1, shuffle=False))

    #pool = Pool(23)
    #partialF =functools.partial(nearest_help, trainset=trainset, dset=dset,writer=writer)
    #pool.map(partialF, list(enumerate(distance_matrix)))

    # writer.add_figure("graph"+ dataset+str(i), get_molecule_plot(dset()))
    for i, l in enumerate(tqdm(distance_matrix)):
        writer.add_figure("graph/"+str(i)+"-", get_molecule_plot(dset[i],**kwargs))
        bests = np.argsort(l)[:10]#.argsortt[:10]
        for j, el in enumerate(bests):
            writer.add_figure("graph/" +str(i)+"-" +str(j), get_molecule_plot(trainset[el], **kwargs))

def nearest_tag(run, dataset, distance_matrix, **kwargs):
    writer = SummaryWriter("runs/distance "+run+" "+kwargs["name"]+"/"+ dataset)

    trainset=list(Dl(kwargs["trainset"].dataset, batch_size=1, shuffle=False))

    dset=list(Dl(kwargs[dataset].dataset, batch_size=1, shuffle=False))

    #pool = Pool(23)
    #partialF =functools.partial(nearest_help, trainset=trainset, dset=dset,writer=writer)
    #pool.map(partialF, list(enumerate(distance_matrix)))

    # writer.add_figure("graph"+ dataset+str(i), get_molecule_plot(dset()))
    tag_list = ["g/" for i, el in enumerate(trainset)]
    for i, l in enumerate(tqdm(distance_matrix)):
        #writer.add_figure("graph/"+str(i)+"-", get_molecule_plot(dset[i]))
        bests = np.argsort(l)[:10]#.argsortt[:10]
        for j, el in enumerate(bests):
            tag_list[el]+="id_"+str(i)+ "_r_"+str(j+1)+"_"
            #writer.add_figure("graph/" +str(i)+"-" +str(j), get_molecule_plot(trainset[el]))
    for i, l in enumerate(tqdm(trainset)):
        writer.add_figure(tag_list[i], get_molecule_plot(l,**kwargs),)
    for i, l in enumerate(tqdm(dset)):
        writer.add_figure("g/id_"+str(i)+"_r_0_", get_molecule_plot(l,**kwargs))


def nearest_help(el, trainset, dset, writer):
    i, l = el
    writer.add_figure("graph/"+str(i)+"_", get_molecule_plot(dset[i]))
    bests = np.argsort(l)[:10]#.argsortt[:10]
    for j, el in enumerate(bests):
        writer.add_figure("graph/" +str(i)+"_" +str(j), get_molecule_plot(trainset[el]))


from distance import  cumulative_match_caracteristic
def plot_cmc(run, dataset, distance_matrix, **kwargs):

    writer = SummaryWriter("runs/cmc/ "+run+" "+kwargs["name"]+"/"+ dataset)
    #train_classes = np.concatenate([el.y for el in kwargs["trainset"]])
    classes = np.concatenate([el.y for el in kwargs["trainset"]])

    cmc = cumulative_match_caracteristic(distance_matrix, classes)
    rcmc =cumulative_match_caracteristic(distance_matrix, classes, neg=1)
    i=0
    while cmc[i]<1:
        writer.add_scalar("cmc", cmc[i], i)
        i+=1
    writer.add_scalar("cmc", cmc[i], i)

    i=0
    while rcmc[i]<1:
        writer.add_scalar("rcmc", rcmc[i], i)
        i+=1
    writer.add_scalar("rcmc", rcmc[i], i)

def plot_random_cmc():
    writer = SummaryWriter("runs/cmc/random")
    for i in range(20):
        writer.add_scalar("cmc", 1-1./2**(1+i), i)
        writer.add_scalar("rcmc", 1-1./2**(1+i), i)


def plot_neigbourhood(dataset, id, atom, radius):
    writer = SummaryWriter("runs/cmc/random")
    for i in range(20):
        writer.add_scalar("cmc", 1-1./2**(1+i), i)
        writer.add_scalar("rcmc", 1-1./2**(1+i), i)


