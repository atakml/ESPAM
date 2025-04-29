
import numpy as np
import os
import subprocess

import networkx as nx

def read_subgroup(filename, **kwargs):
    out = (list(), list(),list())
    with open(filename, "r") as f:
        l = f.readline()
        while True:
            while l[0]!="=":
                l = f.readline()
                if not l:
                    return out

            g =parse_subgraph(f,**kwargs)
            out[2].append(g)
            while l[0] !="P":
                l = f.readline()
            tmp= list()
            l = f.readline()
            while l != "\n":
                tmp.append(parse_line(l))
                l = f.readline()
            out[1].append(tmp)
            while l[0] !="N":
                l = f.readline()
            tmp= list()
            l = f.readline()

            while l != "\n":
                tmp.append(parse_line(l))
                l = f.readline()
            out[0].append(tmp)

            #while l!="\n":

    return out


def parse_line(line):
    mol, nodes = line.split(": ")
    mol= int(mol)
    nodes = nodes.split(" ")
    nodes = [int(n) for n in nodes]
    return mol,nodes

def parse_subgraph(fp, **kwargs):
    l= fp.readline()
    lines = []
    while l!= "\n":
        node, edges = l.split(" : ")
        node, type = node.split("[")
        edges = [int(el.split("(")[0]) for el in  edges.split(" ")]
        lines.append(((int(node), int(type[:-1])), edges))
        l = fp.readline()
    nodes = [(el, {"x":kwargs["atoms"][cl]}) for (el,cl),_ in lines]
    edges = [(el[0], nb) for el,nbs in lines for nb in nbs]
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g.to_undirected()


def run_LTS(graphs, npositives, **kwargs):

    node_data = list()
    edge_data = list()
    for i, g in enumerate(graphs):
        for j in g.nodes():
            """if kwargs["name"][:3]=="syn":
                node_data.append(("nn", str(i), str(j), str(1)))#int(g.nodes[j]["x"]))))
            else:"""
            node_data.append(("nn", str(i), str(j), str(np.argmax(g.nodes[j]["x"]))))
        for el in g.edges():
            edge_data.append(("nn", str(i), str(el[0]), str(el[1]), str(0)))
    dirname= "datasets/subgraphs"
    with open(os.path.join(dirname,"node_data.txt"), "w") as f:
        for l in node_data:
            f.write(" ".join(l)+"\n")
    with open(os.path.join(dirname, "edge_data.txt"), "w") as f:
        for l in edge_data:
            f.write(" ".join(l)+"\n")
    i=0
    bashCommand = "java -Xmx10024m -jar LTS/LTS.jar -n " + os.path.join(dirname,"node_data.txt") + \
                  " -e " +os.path.join(dirname, "edge_data.txt")+" -o "+ os.path.join(dirname, "output") + " -a ltsdfs -p " + str(npositives)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)
    print(error)