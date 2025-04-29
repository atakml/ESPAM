from tqdm import tqdm
import pandas as pd

from ESPAMS import second_degree_ESPAM_on_masks, second_degree_ESPAM_with_BRICS, espam_ego_networks_average
from loaders import load_and_process_data_for_protein, save_to_pkl
from sme.MaskGNN_interpretation.build_data import build_mol_graph_data_for_murcko
from sme.MaskGNN_interpretation.build_data import build_mol_graph_data_for_fg

from mitihelper import from_smiles_to_data


def exp_espam(uni_protid, model, method):
    data = load_and_process_data_for_protein(uni_protid)
    res = []
    for i, row in tqdm(data.iterrows()):
        predicted_class = model(row['mols'], row['prot'])[0].argmax().item()

        if method in ["fg", "sc"]:
            d = pd.DataFrame([[row.mols, predicted_class, "test"]], columns=["smiles", "label", "group"])
            fragment_func = build_mol_graph_data_for_fg if method == "fg" else build_mol_graph_data_for_murcko
            q = fragment_func(d, "label", "smiles", None)
            if not len(q):
                continue
            t = second_degree_ESPAM_on_masks(row['mols'], row['prot'], model, predicted_class, q)
        elif method == "brics":
            t = second_degree_ESPAM_with_BRICS(row['mols'], row['prot'], model, predicted_class)
        elif method == "atom":
            data = from_smiles_to_data(row['mols'])
            t = espam_ego_networks_average(data, row['prot'], model, predicted_class)
        else:
            raise NotImplementedError
        res.append((row['mols'], t))

    save_to_pkl(res, f"ESPAM_{method}_{uni_protid}")
    return res

