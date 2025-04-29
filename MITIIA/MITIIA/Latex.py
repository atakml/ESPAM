import pandas as pd
import pickle
import configs
from utils import convert_conjunction_to_dict, get_support_indices, convert_subgroup_to_dict, build_dataset_from_file, \
    load_mol_dataset
from GNN_Transformer.optimal_transport_for_gnn.src.OTBFS import evaluate_score_from_file


def latex_rule_information(rule_file, data_file, dataset_file):
    """
    :param rule_file: The file containing the rule files
    :param data_file: the file containing activation_matrix, graph_indices and labels. This file is equivalent to
    configs.file_to_read_data
    :param dataset_file: dataset file (predict.csv)
    :return: latex code of the table of the rules
    """
    rules = pd.read_csv(rule_file)
    with open(data_file, "rb") as pkl:
        activation_matrix, graph_indices, labels = pickle.load(pkl)
    dataset = pd.read_csv(dataset_file, delimiter=";")
    s = ""
    for rl in rules.iterrows():
        rule = rl[1]
        t = f"\\{set(sorted(convert_subgroup_to_dict(rule.Rule).keys()))}"
        t = t[:-1] + "\\" + t[-1]
        s = s + t + "&" + str(rule.SI) + "&" + str(rule.Positive_Support) + "&" + str(
            rule.Negative_Support) + "&" + str(rule.Total_Support)
        support_indices = get_support_indices(convert_subgroup_to_dict(rule.Rule), activation_matrix.to_numpy(),
                                              graph_indices)
        support_rows = dataset.iloc[support_indices]
        distinct_seqs = support_rows['seq_id'].nunique()
        s += f"&{distinct_seqs}"
        distinct_graphs = support_rows["_SMILES"].nunique()
        s += f"&{distinct_graphs}"
        dataset_positive = dataset.loc[dataset['pred'] >= 0.5].shape[0]
        dataset_size = dataset.shape[0]
        score = rule.Positive_Support / dataset_positive - rule.Total_Support / dataset_size
        s += f"&{score}\n" + "\\" + "\\" + "\\hline\n"
    return s


def latex_rule_graph_scores(rule_file, bert_func, layer, target=1):
    metric = "cosine"
    rules = pd.read_csv(rule_file)
    file_to_read_data = f"GNN_Transformer/preprocessed_data_{layer}.pkl"
    with open(file_to_read_data, "rb") as pkl:
        activation_matrix, graph_indices, labels = pickle.load(pkl)
    s = ""
    for rl in rules.iterrows():
        rule = rl[1].Rule
        suffix = f"_{convert_subgroup_to_dict(rule)}_{layer}"
        graph_median_file = f"median_graphs/graph_median{suffix}.pkl"
        graph_median_final_file = f"representations/final_ot{suffix}.pkl"
        graph_random_file = f"random_graphs/graph_random{suffix}.pkl"
        final_random_file = f"representations/final_random{suffix}.pkl"
        median_score = evaluate_score_from_file(configs.config_file, graph_median_file, target, convert_subgroup_to_dict(rule),
                                                layer,
                                                activation_matrix.to_numpy(),
                                                metric, bert_func, configs.prediction_file, graph_indices)
        random_score = evaluate_score_from_file(configs.config_file, graph_random_file, target, convert_subgroup_to_dict(rule),
                                                layer,
                                                activation_matrix.to_numpy(),
                                                metric, bert_func, configs.prediction_file, graph_indices)
        with open(graph_median_final_file, "rb") as pkl:
            median_final_score = pickle.load(pkl)[1]
        with open(final_random_file, "rb") as pkl:
            random_final_score = pickle.load(pkl)[1]
        s += f"{median_score}& {median_final_score}& {random_score} &{random_final_score}" + "\n" + "\\" + "\\" + "\\hline" + "\n"
    return s
    
'''
for layer in range(5):
    file_to_read_data = f"GNN_Transformer/preprocessed_data_{layer}.pkl"
    print("-" * 100)
    print(latex_rule_information(f"/mnt/MITIIA/GNN_Transformer/rules/rules_si_divided_10_{layer}.csv", file_to_read_data,
                                 configs.prediction_file))'''

dataset, bert_func = load_mol_dataset(configs.prediction_file, configs.seq_file, configs.bert_file)
for layer in range(5):
    print("-" * 100)
    print(latex_rule_graph_scores(f"/mnt/MITIIA/GNN_Transformer/rules/rules_si_divided_10_{layer}.csv", bert_func, layer))
