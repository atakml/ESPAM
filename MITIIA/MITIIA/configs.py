random = False  # whether to use random or graph median for starting graph

target = 1
layer = 0  
input_seq_id = None
number_of_rules = 10
file_to_write_rules = f"GNN_Transformer/rules/rules_{number_of_rules}_{layer}.csv"  # discovered rules by beamsearch
intermediate_file = "GNN_Transformer/intermediates.pkl"  # Embedding of all vertices and all layers of GNN
prediction_file = "GNN_Transformer/predict.csv"  # This file is the summary of the dataset_file and it is used to load mols
file_to_read_data = f"GNN_Transformer/preprocessed_data_{layer}.pkl"  # The file that contains activation matrix, graph indices and labels
dataset_file = "GNN_Transformer/data_train_node32_edge64.csv"  # It's the file that have been used to generate the activation patterns
seq_file = "GNN_Transformer/seqs.csv"  # The file that is needed to convert seq_id to seq
bert_file = "GNN_Transformer/BERT_GNN/Data/chemosimdb/mixOnly_20220621-145302/PrecomputeProtBERT_CLS/ProtBERT_CLS.h5"
final_file_format = "representations/final_ot_{rule}_{layer}.pkl" if not random else "representations/final_random_{rule}_{layer}.pkl"  # The file to save to final graph representation
config_file = "GNN_Transformer/configs/config_template.yml"
file_to_write_support_graphs_format = "GNN_Transformer/rule_supports/support{rule}_{layer}.json"
start_graph_format = "median_graphs/graph_median_{rule}_{layer}.pkl" if not random else "random_graphs/graph_random_{rule}_{layer}.pkl"
suffix_format = "median_graphs/graph_median_{rule}_{layer}" if not random else "random_graphs/graph_random_{rule}_{layer}"  # is used in pars_json file

hparams = {"ATOM_FEATURES": ["AtomicNum", "ChiralTag", "Hybridization", "FormalCharge", "NumImplicitHs", "ExplicitValence", "Mass", "IsAromatic"], "BOND_FEATURES": ["BondType", "Stereo", "IsAromatic"]}
