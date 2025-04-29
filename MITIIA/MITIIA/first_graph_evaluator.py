from configs import *
from GNN_Transformer.optimal_transport_for_gnn.src.OTBFS import evaluate_score_from_file
from utils import build_dataset_from_file, read_from_beam_search_files, write_support_graphs_to_file, load_mol_dataset, \
    filter_by_seq_id
from tqdm import tqdm
file_ot = "/mnt/MITIIA/ot_start.json"
file_random = "/mnt/MITIIA/random_start.json"
dataset, bert_func = load_mol_dataset(prediction_file, seq_file, bert_file)
file_to_write_rules = f"GNN_Transformer/rules.csv"
rules = read_from_beam_search_files(file_to_write_rules)
for rule in tqdm(rules.iterrows()):

    print("ot:")
    evaluate_score_from_file("GNN_Transformer/configs/config_template.yml", file_ot, target,
                                   rule[1]['Rule'], layer, activation_matrix.to_numpy(), "cosine", bert_func,
                                   prediction_file, graph_indices)

    print("random")
    evaluate_score_from_file("GNN_Transformer/configs/config_template.yml", file_ot, target,
                                   rule[1]['Rule'], layer, activation_matrix.to_numpy(), "cosine", bert_func,
                                   prediction_file, graph_indices)

