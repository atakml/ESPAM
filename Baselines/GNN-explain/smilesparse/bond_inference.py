import networkx as nx

from smilesparse.visitsmiles import VisitSmiles

link_count = {'C': [4],
              'N': [3, 5],
              'O': [2],
              'S': [2, 4, 6],
              'P': [3, 5],
              "BR": [1],
              "B": [3],
              "F": [1],
              "CL": [1],
              "I": [1]}
bond_count = {'': 1, '-': 1, '\\': 1, '/': 1, '=': 2, '#': 3, ".": 0, ":": 2}


class TooMuchBonds(Exception):
    pass


def bond_inference(graph: nx.Graph, allow_overbonds=False, isometry=[]):
    #graph = hydrogen_fill(graph, allow_overbonds)
    if "ze" in isometry:
        graph = ze_isometry(graph)
    return graph


def hydrogen_fill(graph: nx.Graph, allow_overbonds=False):
    """
    fill atoms with hydrogens to match the number of covalent bonds
    :param allow_overbonds:
    :param graph:
    :return:
    """
    node_list = list(graph.nodes)
    for el in node_list:
        if graph.nodes[el]["h_count"] is None:
            count = 0
            for n in graph.neighbors(el):
                bond = graph.get_edge_data(el, n)["bond"]
                count += bond_count[bond]
            num_link = link_count[graph.nodes[el]["symbol"].upper()]
            missing_bonds = 0
            for i in num_link:
                missing_bonds = i - count
                if missing_bonds > 0:
                    break
            if missing_bonds < 0 and not allow_overbonds:
                raise TooMuchBonds(
                    "atom : %d of type %s can accept at most %d bonds\nBut %d provided\n".format(
                        el, graph.nodes[el]["symbol"], num_link[-1], count))

            for i in range(missing_bonds):
                new_node_index = graph.number_of_nodes() + 1
                graph.add_node(new_node_index, **VisitSmiles.H_LABEL)
                graph.add_edge(el, new_node_index, bond="-")
    return graph


def ze_isometry(graph: nx.Graph):
    """
    manages Z/E isometry by adding special edges
    :param graph:
    :return:
    """
    print("Z/E isometry Not implemented")
    return graph
