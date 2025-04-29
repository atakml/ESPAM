import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx

from dataloader import atoms


def plot_edonicity(*molecules,number=221):
    plt.figure(figsize=[24,16])
    for i, (el,indexes, mol_values) in enumerate(molecules):
        mol= to_networkx(el,node_attrs="x").to_undirected()
        labels= {node: atoms[el.argmax()]#"%1.2f"%float(mol_values[i][0][1].sum()))
                 for i,(node, el) in enumerate(zip(mol.nodes(), el.x)) if atoms[el.argmax()]!="H"}
        nodes ={node for node, el in zip(mol.nodes(), el.x) if atoms[el.argmax()]!="H"}
        plt.subplot(number+i)
        nx.draw_kamada_kawai(mol, node_size=1, with_labels=False)
        nx.draw_kamada_kawai(mol, nodelist=nodes, labels=labels, with_labels=True)
        nx.draw_kamada_kawai(mol, nodelist=indexes, node_color="r", labels=labels, with_labels=False)

    plt.show()

def plot_sal(*molecules):
    plt.figure(figsize=[30,20])
    for i, (el,values) in enumerate(molecules):
        mol= to_networkx(el,node_attrs="x").to_undirected()
        labels= {node: "{:.2e}".format(val)#"%1.2f"%float(mol_values[i][0][1].sum()))
                 for node, val in zip(mol.nodes(), values)}
        nodes ={node for node, el in zip(mol.nodes(), el.x)}
        plt.subplot(231+i)
        #nx.draw_kamada_kawai(mol, node_size=5, with_labels=False)
        nx.draw_kamada_kawai(mol,node_size=10, nodelist=nodes, labels=labels, with_labels=True)
        #nx.draw_kamada_kawai(mol, nodelist=indexes, node_color="r", labels=labels, with_labels=False)

    plt.show()

def plot_grad(*molecules):
    plt.figure(figsize=[30,20])
    for i, (el, values) in enumerate(molecules):
        mol = to_networkx(el,node_attrs="x").to_undirected()
        labels= {node: "{:.2e}".format(val.sum().to(float))#"%1.2f"%float(mol_values[i][0][1].sum()))
                 for node, val in zip(mol.nodes(), values)}
        nodes ={node for node, el in zip(mol.nodes(), el.x)}
        plt.subplot(111)
        #nx.draw_kamada_kawai(mol, node_size=5, with_labels=False)
        nx.draw_kamada_kawai(mol,node_size=10, nodelist=nodes, labels=labels, with_labels=True)
        #nx.draw_kamada_kawai(mol, nodelist=indexes, node_color="r", labels=labels, with_labels=False)

    plt.show()
    """
        for mol, expl in zip(kwargs["trainset"].dataset, explanations):
            print(expl.sum())
            #plot_molecule.plot_grad(*[(mol,expl)])
    """

def plot_mol(size, molecules):
    plt.figure(figsize=[4*size[1],4*size[0]])
    for i, el in enumerate(molecules):
        mol= to_networkx(el,node_attrs="x").to_undirected()
        labels= {node: atoms[el.argmax()]#"%1.2f"%float(mol_values[i][0][1].sum()))
                 for i,(node, el) in enumerate(zip(mol.nodes(), el.x)) if atoms[el.argmax()]!="H"}
        nodes ={node for node, el in zip(mol.nodes(), el.x) if atoms[el.argmax()]!="H"}
        plt.subplot(size[0],size[1], i+1)
        nx.draw_kamada_kawai(mol, node_size=1, with_labels=False)
        nx.draw_kamada_kawai(mol, nodelist=nodes, labels=labels, with_labels=True)

    plt.show()



import numpy as np
def plot_rules(data,mean=0.5):
    plt.figure(figsize=[24,16])
    colors =  ['b', 'g', 'r', 'y', 'k', 'w']
    markers = ['+', 'x', '.', '2', '3', '1']
    layers=max([el[2] for el in data])
    mx =0

    for i in range(layers):
        layer = [(a,b) for (a,b,c) in data if c==i]
        x = [el[0] for el in layer]
        y = [el[1] for el in layer]
        mx = max(x+[mx])

        plt.scatter(x,y,marker=markers[i],color=colors[i])

    x = np.linspace(1, mx, 10000)

    y = x * (1 - mean) / mean
    plt.plot(x, y)
    coef =1.96
    err = coef*np.sqrt(mean*(1-mean)/(y+x))
    yp = x/(mean-err)-x
    ym = x/(mean+err)-x

    plt.plot(x[200:], yp[200:])  # on utilise la fonction sinus de Numpy
    plt.plot(x[200:], ym[200:])  # on utilise la fonction sinus de Numpy

    #plt.plot(x, y+err)  # on utilise la fonction sinus de Numpy
    #plt.plot(x, 1./mean*x+2*np.sqrt(mean(1-mean)))  # on utilise la fonction sinus de Numpy

    plt.legend(['mean','err+','err-']+[str(el) for el in range(layers)])
    plt.xlabel("number of positive elements")
    plt.ylabel("number of negative elemnts")

    plt.show()

