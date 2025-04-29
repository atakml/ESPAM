from transformers import BertConfig, FlaxBertModel

from utils import build_dataset_from_file, read_from_beam_search_files, write_support_graphs_to_file, load_mol_dataset, \
    filter_by_seq_id
from GNN_Transformer.MaxEntCalculator import build_model
from beamsearch import beam_search, Data
import os
import pickle
from GNN_Transformer.optimal_transport_for_gnn.src.parse_json import median_from_json
from GNN_Transformer.optimal_transport_for_gnn.src.OTBFS import best_first_search_json
import code
from tqdm import tqdm
from configs import *

dataset, bert_func = load_mol_dataset(prediction_file, seq_file, bert_file)
print("Building dataset ...", end=" ", flush=True)
if os.path.exists(file_to_read_data):
    with open(file_to_read_data, "rb") as pkl:
        activation_matrix, graph_indices, labels = pickle.load(pkl)
else:
    activation_matrix, graph_indices, labels = build_dataset_from_file(intermediate_file, prediction_file, layer=layer)
    with open(file_to_read_data, "wb") as pkl:
        pickle.dump((activation_matrix, graph_indices, labels), pkl)
if input_seq_id is not None:
    dataset, activation_matrix, graph_indices, labels = filter_by_seq_id(dataset, activation_matrix, graph_indices,
                                                                         labels,
                                                                         input_seq_id)
print("done.")

if not os.path.exists(file_to_write_rules):
    print("Building MaxEnt ...", end=" ", flush=True)
    probabilistic_model = build_model(activation_matrix)
    print("done.")
    data = Data(probabilistic_model, activation_matrix, graph_indices, labels)
    print("Beam search...", end=" ", flush=True)
    beam_search(data, target, file_to_write_rules, max_rules=10)
    print("done.")
print("!!")
rules = read_from_beam_search_files(file_to_write_rules)
print("!!!")
for rule in tqdm(rules.iterrows()):
    print("??")
    file_to_write_support_graphs = file_to_write_support_graphs_format.format(rule=str(rule[1].Rule), layer=layer)
    write_support_graphs_to_file(rule[1], activation_matrix.to_numpy(), graph_indices, dataset,
                                 file_to_write_support_graphs)
    print("???")
    start_graph = start_graph_format.format(rule=str(rule[1].Rule), layer=layer)
    suffix = suffix_format.format(rule=str(rule[1].Rule), layer=layer)
    if not os.path.exists(f"/mnt/MITIIA/{start_graph}"):
        median_from_json("/mnt/MITIIA/GNN_Transformer/", f"rule_supports/support{str(rule[1].Rule)}_{layer}.json", suffix, bert_func,
                         activation_matrix.to_numpy(), graph_indices, rule, 6 * (1 + layer), random)

    final_file = final_file_format.format(rule=str(rule[1].Rule), layer=layer)
    with open(final_file, "ab") as pkl:
        pickle.dump(
            best_first_search_json("GNN_Transformer/configs/config_template.yml", start_graph, target, rule[1]['Rule'],
                                   layer, activation_matrix.to_numpy(), "cosine", bert_func, prediction_file,
                                   graph_indices), pkl)
