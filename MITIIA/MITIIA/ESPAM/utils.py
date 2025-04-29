from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.typing import OptTensor, PairTensor

from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes
from rdkit.Chem.rdmolops import DeleteSubstructs
from rdkit import Chem
import datamol as dm
from functools import partial, reduce

import dgl
from torch_geometric.utils import k_hop_subgraph, mask_to_index, index_to_mask
import torch_geometric.transforms as T
from torch_geometric.data import Data

from ESPAM.mitihelper import from_smiles_to_data, compute_model_score
# ESPAM1
from torch_geometric.utils import remove_self_loops, k_hop_subgraph
from pprint import pprint


def k_hop_subgraph1(
        node_idx: Union[int, List[int], Tensor],
        num_hops: int,
        edge_index: Tensor,
        relabel_nodes: bool = False,
        num_nodes: Optional[int] = None,
        flow: str = 'source_to_target',
        directed: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Computes the induced subgraph of :obj:`edge_index` around all nodes in
    :attr:`node_idx` reachable within :math:`k` hops.

    The :attr:`flow` argument denotes the direction of edges for finding
    :math:`k`-hop neighbors. If set to :obj:`"source_to_target"`, then the
    method will find all neighbors that point to the initial set of seed nodes
    in :attr:`node_idx.`
    This mimics the natural flow of message passing in Graph Neural Networks.

    The method returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central seed
            node(s).
        num_hops (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (str, optional): The flow direction of :math:`k`-hop aggregation
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
        ...                            [2, 2, 4, 4, 6, 6]])

        >>> # Center node 6, 2-hops
        >>> subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        ...     6, 2, edge_index, relabel_nodes=True)
        >>> subset
        tensor([2, 3, 4, 5, 6])
        >>> edge_index
        tensor([[0, 1, 2, 3],
                [2, 2, 4, 4]])
        >>> mapping
        tensor([4])
        >>> edge_mask
        tensor([False, False,  True,  True,  True,  True])
        >>> subset[mapping]
        tensor([6])

        >>> edge_index = torch.tensor([[1, 2, 4, 5],
        ...                            [0, 1, 5, 6]])
        >>> (subset, edge_index,
        ...  mapping, edge_mask) = k_hop_subgraph([0, 6], 2,
        ...                                       edge_index,
        ...                                       relabel_nodes=True)
        >>> subset
        tensor([0, 1, 2, 4, 5, 6])
        >>> edge_index
        tensor([[1, 2, 3, 4],
                [0, 1, 4, 5]])
        >>> mapping
        tensor([0, 5])
        >>> edge_mask
        tensor([True, True, True, True])
        >>> subset[mapping]
        tensor([0, 6])
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if isinstance(node_idx, list):
        num_nodes = max(num_nodes, max(node_idx) + 1)
    else:
        num_nodes = max(num_nodes, node_idx + 1)
    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]
    tmp_subset = []
    for i in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        if (i < num_hops - 1):  ### j'ai ajoute ce test pour enlever la derniere couronne de noeuds
            subsets.append(col[edge_mask])
        else:
            tmp_subset.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True, sorted=False)
    node_mask.fill_(False)
    node_mask[subset] = True

    if not directed:
        edge_mask = node_mask[row] | node_mask[col]  ## j'ai change le & en |
        if num_hops == 0:
            edge_mask = node_mask[row] & node_mask[col]
        # il suffit qu'un noeud adjacent ait ete atteind pour qu'on enleve l'arete (et non pas les deux comme avant)

    edge_index = edge_index[:, edge_mask]

    if len(tmp_subset):
        subsets.append(tmp_subset[0])
    subset, inv = torch.cat(subsets).unique(return_inverse=True, sorted=False)
    inv = inv[:node_idx.numel()]
    node_mask[subset] = True
    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]
    return subset, edge_index, inv, edge_mask


def remove_redundancy(mol):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    pattern = Chem.MolFromSmiles('*')
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return Chem.Mol(mol)
    res = mol
    # while matches:
    #    match = matches[0]
    #    res = Chem.RWMol(res)
    #    res.BeginBatchEdit()
    #    for aid in match:
    #        res.RemoveAtom(aid)
    #    res.CommitBatchEdit()
    #    Chem.SanitizeMol(res)
    #    matches = res.GetSubstructMatches(pattern)
    # return res
    # print(f"mol: {Chem.MolToSmiles(mol)}")
    res = dm.remove_dummies(mol)
    # print(f"res: {Chem.MolToSmiles(res)}")
    # print("end: remove_redundancy")
    return res


def remove_fragment(mol, fragment, match_id):
    if not isinstance(mol, str):
        mol = Chem.MolToSmiles(mol)
    mol_data = from_smiles_to_data(mol)
    match = list(reduce(lambda x, y: x + y, match_id)) if len(match_id) > 1 else match_id
    mol_data.x[match, :] = torch.zeros(mol_data.x.shape[1])
    node_mask = torch.zeros(mol_data.x.shape[0])
    node_mask[match] = 1
    node_mask = node_mask.bool()
    edge_mask = node_mask[mol_data.edge_index[0]] | node_mask[mol_data.edge_index[1]]
    mol_data.edge_index = mol_data.edge_index[:, ~edge_mask]
    mol_data.edge_attr = mol_data.edge_attr[~edge_mask, :]
    return mol_data


def features_to_graph(data, z, l):
    center_nodes = []
    for i in range(len(z)):
        if z[i] == 0:
            center_nodes.append(i)

    kh_subset, kh_edge_index, kh_mapping, kh_edge_mask = k_hop_subgraph1(center_nodes, l, data.edge_index,
                                                                         relabel_nodes=False)
    g = dgl.graph((data.edge_index[0], data.edge_index[1]))
    g.edata['x'] = data.edge_attr
    g.remove_edges(mask_to_index(kh_edge_mask))

    edges_src, edges_dst = g.edges()
    edge_index = torch.tensor([edges_src.numpy(), edges_dst.numpy()], dtype=torch.long)
    data = Data(x=data.x, edge_index=edge_index, y=data.y, shape=data.shape, edge_attr=g.edata['x'])

    return data


def relable_nodes(feature, edge_index, node_subset):
    node_subset = torch.unique(node_subset, return_inverse=True, sorted=True)
    node_dict = {node_subset[0][i].item(): node_subset[1][i].item() for i in range(node_subset[0].shape[0])}
    edge_index = torch.tensor([[node_dict[u.item()], node_dict[v.item()]] for u, v in edge_index.T]).T
    return feature[node_subset[0]], edge_index
