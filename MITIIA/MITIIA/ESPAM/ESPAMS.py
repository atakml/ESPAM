from rdkit.Chem.rdmolops import DeleteSubstructs
from rdkit import Chem
from rdkit.Chem.BRICS import BRICSDecompose, BRICSBuild
from rdkit.Chem import RDConfig
import os
import sys
import torch
from torch_geometric.data import Data
from mol2graph.jraph.convert import smiles_to_jraph


sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from functools import partial, reduce
from tqdm import tqdm
from ESPAM.mitihelper import compute_model_score, from_smiles_to_data
from ESPAM.utils import remove_fragment, k_hop_subgraph1, relable_nodes, features_to_graph, remove_redundancy

def compute_second_degree_espam_score(fragments_score, combined_fragment_score, combined_fragment_complement_score,
                                      fragment_complements_score, w1, w2):
    contributions = []
    for i in range(len(fragments_score)):
        contributions.append(w1 * fragments_score[i] + w2 * sum(combined_fragment_score[i]) - (1 - w2) * sum(
            combined_fragment_complement_score[i]) - (1 - w1) * fragment_complements_score[i])
    return contributions


def second_degree_ESPAM_with_BRICS(mol, seq, model, target_class):
    score = partial(compute_model_score, seq=seq, model=model, class_index=target_class)
    data = from_smiles_to_data(mol)
    m = Chem.MolFromSmiles(mol)
    fragments_smiles = list(Chem.BRICS.BRICSDecompose(m))
    fragments_matches = list(
        map(lambda x: list(map(lambda y: (x, y), m.GetSubstructMatches(remove_redundancy(x)))), fragments_smiles))
    fragments_matches = list(reduce(lambda x, y: x + y, fragments_matches))
    #fragments_matches = list(map(lambda x: (smiles_to_jraph(x[0], atom_features=["AtomicNum", "ChiralTag", "Hybridization", "FormalCharge", "NumImplicitHs", "ExplicitValence", "Mass", "IsAromatic"], bond_features=["BondType", "Stereo", "IsAromatic"]), x[1]), fragments_matches))
    print(len(fragments_smiles))
    fragments_combinations = []
    fragments_score = []
    fragments_combinations_score = []
    fragment_complements = []
    fragment_complements_score = []
    combined_fragment_complement = []
    combined_fragment_complement_score = []
    combined_fragment_score = []
    #fragments_score = model([(seq, mol, 1) for mol, _ in fragments_matches])
    for i in range(len(fragments_matches)):
        fragments_score.append(score(fragments_matches[i][0]))
        fragment_complements.append(remove_fragment(m, fragments_matches[i][0], [fragments_matches[i][1]]))
        sc = compute_model_score(fragment_complements[-1], seq, model, target_class)
        fragment_complements_score.append(sc)
        combined_fragment_complement.append([])
        combined_fragment_score.append([])
        fragments_combinations.append([])
        fragments_combinations_score.append([])
        for j in range(len(fragments_matches)):
            if i == j:
                continue
            combined_fragment_complement[-1].append(
                remove_fragment(m, fragments_matches[j][0], (fragments_matches[i][1], fragments_matches[j][1])))
        combined_fragment_complement_score.append(list(map(lambda x: score(x), combined_fragment_complement[-1])))
        for j in range(len(fragments_matches)):
            if j == i:
                continue
            nodes_to_keep = [
                list(set(range(data.x.shape[0])) - set((fragments_matches[i][1] + fragments_matches[j][1])))]
            fragments_combinations[-1].append(remove_fragment(m, [], nodes_to_keep))
            fragments_combinations_score[-1].append(score(fragments_combinations[-1][-1]))
    w1 = (score(from_smiles_to_data(mol)) / 2 + sum(fragment_complements_score)) / (
            sum(fragment_complements_score) + sum(fragments_score))
    w2 = (score(from_smiles_to_data(mol)) / 2 + sum(list(map(sum, combined_fragment_complement_score)))) / (
            sum(list(map(sum, combined_fragment_complement_score))) + sum(list(map(sum, combined_fragment_score))))
    contributions = compute_second_degree_espam_score(fragments_score, combined_fragment_score,
                                                      combined_fragment_complement_score, fragment_complements_score,
                                                      w1, w2)
    return fragments_matches, contributions


def second_degree_ESPAM_on_masks(smiles, seq, model, target_class, mol_with_mask):
    # used on functional groups and scaffolds
    assert len(mol_with_mask) > 0, smiles
    score = partial(compute_model_score, seq=seq, model=model, class_index=target_class)
    data = from_smiles_to_data(smiles)
    mol = smiles
    if len(mol_with_mask) == 1:
        contributions = [score(smiles)]
    else:
        complement_scores = list(map(lambda x: score(remove_fragment(x[0], [], [x[4]])), mol_with_mask))
        node_set = set(list(range(data.x.shape[0])))
        single_graphs = list(map(lambda x: remove_fragment(x[0], [], [list(node_set - set(x[4]))]), mol_with_mask))
        single_scores = list(map(lambda x: score(x), single_graphs))
        combined_scores = []
        combined_complement_scores = []
        for i in range(len(mol_with_mask)):
            combined_scores.append([])
            combined_complement_scores.append([])
            for j in range(len(mol_with_mask)):
                if i == j:
                    continue
                combined_complement_graph = remove_fragment(mol, [], [mol_with_mask[i][4], mol_with_mask[j][4]])
                combined_complement_scores[-1].append(score(combined_complement_graph))
                combined_graph = remove_fragment(mol, [],
                                                 [list(node_set - set(mol_with_mask[i][4] + mol_with_mask[j][4]))])
                combined_scores[-1].append(score(combined_graph))
        mol_score = score(smiles)
        w1 = (mol_score / 2 + sum(complement_scores)) / (sum(complement_scores) + sum(single_scores))
        w2 = (mol_score / 2 + sum(list(map(sum, combined_complement_scores)))) / (
                sum(list(map(sum, combined_complement_scores))) + sum(list(map(sum, combined_scores))))
        contributions = compute_second_degree_espam_score(single_scores, combined_scores, combined_complement_scores,
                                                          complement_scores, w1, w2)
    return mol_with_mask, contributions


def espam_ego_networks_average(data, seq, model, class_index, ll=4):
    number_of_nodes = data.x.shape[0]
    ego_sets = []
    graph_score = compute_model_score(data, seq, model, class_index)
    ego_contributions = []
    complement_ego_sets = []
    for l in range(1, ll):
        ego_sets.append([k_hop_subgraph1(node_idx, l, data.edge_index) for node_idx in range(number_of_nodes)])
        while ego_sets[-1][-1][1].shape[1] == 0:
            print("!")
            ego_sets[-1].pop()
            number_of_nodes -= 1
        ego_sets[-1] = list(map(lambda x: list(relable_nodes(data.x, x[1], x[0])) + [x[2], x[3]], ego_sets[-1]))
        complement_ego_sets.append(
            [features_to_graph(data, torch.ones(data.x.shape[0]).scatter_(-1, torch.tensor([node_idx]), 0), l) for
             node_idx in range(number_of_nodes)])

        complement_scores = list(
            map(lambda x: compute_model_score(x, seq, model, class_index), complement_ego_sets[-1]))
        complement_sums = sum(complement_scores)
        ego_scores = list(
            map(lambda x: compute_model_score(Data(x=x[0], edge_index=x[1], edge_attr=data.edge_attr[x[3]]), seq, model,
                                              class_index), ego_sets[-1]))
        ego_sums = sum(ego_scores)
        w = (graph_score + complement_sums) / (complement_sums + ego_sums)
        ego_contributions.append(list(
            map(lambda node_index: w * ego_scores[node_index] + (1 - w) * (-complement_scores[node_index]),
                range(number_of_nodes))))
    node_contributions = torch.tensor(ego_contributions).sum(dim=0) / (ll-1)
    return node_contributions

