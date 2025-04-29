import os
import pickle
import tables
import shutil
import uuid
import argparse
import random
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from jax import numpy as jnp
import jax
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit_heatmaps import mapvalues2mol
from rdkit_heatmaps.utils import transform2png
from more_itertools import set_partitions
from importlib import reload
from ESPAM.ESPAMS import second_degree_ESPAM_with_BRICS
from GNN_Transformer.scripts.main import prepare_params
from GNN_Transformer.main_predict_single import main_predict_single
from GStarX.gstarx import GStarX
import ESPAM.mitihelper as mi
from ESPAM.mitihelper import compute_model_score, from_smiles_to_data

jax.config.update('jax_platform_name', 'cpu')
model_config_file = "GNN_Transformer/configs/config_template.yml"
params = prepare_params(model_config_file)
model = main_predict_single(params)

reload(mi)

def generate_cls(creat=False):
    if creat or not os.path.exists("cls_dict.pkl"):
        cls_dict = {}
        try:
            h5file = tables.open_file('BERT_GNN/Data/chemosimdb/mixOnly_20220621-145302/PrecomputeProtBERT_CLS/ProtBERT_CLS.h5', mode='r', title="TapeBERT")
        except IOError:
            temp_file = f'BERT_GNN/Data/chemosimdb/mixOnly_20220621-145302/PrecomputeProtBERT_CLS/ProtBERT_CLS_copy_{uuid.uuid4().hex}.h5'
            shutil.copy('BERT_GNN/Data/chemosimdb/mixOnly_20220621-145302/PrecomputeProtBERT_CLS/ProtBERT_CLS.h5', temp_file)
            h5file = tables.open_file(temp_file, mode='r', title="TapeBERT")
        bert_table = h5file.root.bert.BERTtable
        for i, seq_id in enumerate(bert_table.col('id')):
            cls_dict[seq_id.astype(str)] = bert_table.col('hidden_states')[i]
        with open("cls_dict.pkl", "wb") as f:
            pickle.dump(cls_dict, f)
    else:
        with open("cls_dict.pkl", "rb") as f:
            cls_dict = pickle.load(f)
    def cls_finder(seq_id):
        return cls_dict[seq_id].reshape(1, -1)
    return cls_finder

bert_func = generate_cls()

def load_ESPAM_data_for_receptor(receptor_name, l=3):
    """
        This function loads the ESPAM data for a given receptor
    """
    with open(f"ESPAM_atom_{receptor_name}_{l}_nice.pkl", "rb") as file:
        res = pickle.load(file)
    return res

def get_smiles_list(ESPAM_data):
    """
        This function returns a list of smiles included in the ESPAM data which is loaded by load_ESPAM_data_for_receptor
    """
    return list(ESPAM_data.keys())

def draw_molecule(smiles, espam, save_name):
    """
        This function takes a SMILES and espam values and a save name to draw the heat map. 
        Please use the visualize function for drawing the heat maps
    """
    mol = Chem.MolFromSmiles(smiles)  
    test_mol = Draw.PrepareMolForDrawing(mol)
    heat_map = np.zeros(test_mol.GetNumAtoms())
    for key, value in enumerate(espam):
        heat_map[key] = value
    canvas = mapvalues2mol(test_mol, heat_map)
    img = transform2png(canvas.GetDrawingText())
    img.save(save_name)

def visualize(smiles, seq_id, ESPAM_data, save_name = None):
    """
        This function takes SMILES, sequence id, ESPAM data, and an optional parameter of the save_name
        If there is no error, the function draws the heat map and ends with the message:
        "Figure saved under the name {save_name}.png" where save name is {smiles}_{seq_id} if no name is given, otherwise the given name
    """
    if save_name is None:
        save_name = f"{smiles}_{seq_id}"
    assert smiles in ESPAM_data.keys(), "SMILES not found."
    assert seq_id in ESPAM_data[smiles].keys(), "There is no such sequence for this molecule."
    try:
        draw_molecule(smiles, ESPAM_data[smiles][seq_id]["ESPAM"], f"{save_name}.png")
    except:
        print("There was an error. Drawing was not successfull.")
        raise
    print(f"Figure saved under the name {save_name}.png")

def get_model_decision(smiles, seq_id, ESPAM_data):
    """
        Returns model decision for a SMILES and a sequence id.
    """
    assert smiles in ESPAM_data.keys(), "SMILES not found."
    assert seq_id in ESPAM_data[smiles].keys(), "There is no such sequence for this molecule."
    return 1 - ESPAM_data[smiles][seq_id]["predicted_class"]

def get_sequences_for_smiles(smiles, ESPAM_data):
    """
        Returns the list of the sequence ids that had interaction with the given smiles.
    """
    assert smiles in ESPAM_data.keys(), "SMILES not found."
    return list(ESPAM_data[smiles].keys())

def get_ESPAM_values(smiles, seq_id, ESPAM_data):
    """
        Return a list of tuples in which the first component is the atom name and the second component is its ESPAM value
        Parameters are SMILES, sequence id and ESPAM_data
    """
    assert smiles in ESPAM_data.keys(), "SMILES not found."
    assert seq_id in ESPAM_data[smiles].keys(), "There is no such sequence for this molecule."
    mol = Chem.MolFromSmiles(smiles)
    return list(zip(list(map(lambda x: x.GetSymbol(), mol.GetAtoms())), ESPAM_data[smiles][seq_id]["ESPAM"]))

def gstarx_model_function(seq_id, model, cls_dict=None):             
    seq = bert_func(seq_id) if cls_dict is None else cls_dict[seq_id]
    def model_output(data):
        x = mi.compute_model_score(data.cpu(), seq, model)
        if isinstance(x, int):
            return torch.tensor([[0, 1]]).cuda()
        return torch.tensor(x.tolist()).cuda()
    return model_output

def main(protein, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    device = "cuda:0"
    gstar_dict = {}
    espam_data = load_ESPAM_data_for_receptor(protein, l=16)
    smiles_list = get_smiles_list(espam_data)
    for smiles in tqdm(smiles_list):
        data = from_smiles_to_data(smiles).cuda()
        data.num_nodes = data.x.shape[0]
        seq_ids = get_sequences_for_smiles(smiles, espam_data)
        for seq_id in seq_ids: 
            gstarx_model = gstarx_model_function(seq_id, model)
            gstarxexplainer = GStarX(gstarx_model, device, payoff_type="raw")
            gstar_dict[(smiles, seq_id)] = gstarxexplainer.explain(data)
    with open(f"GSTARX_res_{protein}_{seed}.pkl", "wb") as file:
        pickle.dump(gstar_dict, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GStarX for a specific protein and seed.")
    parser.add_argument("--protein", type=str, required=True, help="Protein name (e.g., OR1A1, OR5K1, etc.)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed value")
    args = parser.parse_args()
    main(args.protein, args.seed)
