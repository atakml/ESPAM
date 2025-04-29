import pandas as pd
import pickle


def load_and_process_data_for_protein(uni_protid, data_file_name="data_train_node32_edge64.csv", seq_file="seqs.csv",
                                      mol_file="mols.csv"):
    data_file = pd.read_csv(data_file_name, delimiter=";")
    seq_map = pd.read_csv(seq_file, delimiter=";")
    mol_map = pd.read_csv(mol_file, delimiter=";")
    data = data_file.loc[data_file['seq_id'] == uni_protid]
    #smiles_list = data['InChI Key']
    #smiles_list = smiles_list.apply(lambda x: mol_map.loc[mol_map["InChI Key"] == x].head(1)['Isomeric SMILES'].item())
    #data['mols'] = smiles_list
    #data = data[['mols', "UniProt ID", 'class']]
    #data['prot'] = data['UniProt ID'].apply(lambda x: seq_map.loc[seq_map["UniProt ID"] == x].head(1)['seq'].item())
    return data


def pkl_loader(file_name):
    with open(file_name, "rb") as file:
        return pickle.load(file)


def save_to_pkl(data, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(data, file)
