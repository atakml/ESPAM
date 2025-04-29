import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import compute_SI_double, get_support_indices, convert_conjunction_to_dict
import pysubgroup as ps

positive_data, positive_probs, negative_data, negative_probs = None, None, None, None


@dataclass
class Data:
    # Data: a class with these methods: probs (they are log_probs), graph_inds, activation matrix (know as data), labels
    probs: object()
    activation_matrix: object()
    graph_inds: object()
    labels: object()
    positive_probs: object = field(init=False)
    negative_probs: object = field(init=False)
    positive_data: object = field(init=False)
    negative_data: object = field(init=False)
    positive_graph_inds: object = field(init=False)
    negative_graph_inds: object = field(init=False)
    w0: int = field(init=False)
    w1: int = field(init=False)

    def __post_init__(self):
        self.positive_data = self.activation_matrix[self.labels]
        self.positive_graph_inds = self.graph_inds[self.labels]
        self.split_probs()
        self.negative_data = self.activation_matrix[~self.labels]
        self.negative_graph_inds = self.graph_inds[~self.labels]
        n_samples1 = len(np.unique(self.positive_graph_inds))
        n_samples0 = np.max(self.graph_inds) - n_samples1
        self.w0 = max(1, n_samples1 / n_samples0)
        self.w1 = max(1, n_samples0 / n_samples1)

    def split_probs(self):
        self.positive_probs = self.probs[self.labels]
        self.negative_probs = self.probs[~self.labels]

    def __len__(self):
        return len(self.activation_matrix)


class SubjectiveInterestingness:

    def calculate_constant_statistics(self, data):
        pass

    def calculate_statistics(self, data, target):
        pass

    @staticmethod
    def evaluate(subgroup, target, data, statistics_or_data=None):
        pattern = convert_conjunction_to_dict(subgroup)
        res = data.w1 * compute_SI_double(data.positive_probs, data.positive_data.to_numpy(), pattern,
                                          data.positive_graph_inds) - (
                      data.w0 * compute_SI_double(data.negative_probs, data.negative_data.to_numpy(), pattern,
                                                  data.negative_graph_inds))
        if not target.target_selector.attribute_value:
            res *= -1
        return res

    def optimistic_estimate(self, subgroup, statistics=None):
        """ returns optimistic estimate
            if one is available return it otherwise infinity"""
        return math.inf


def delete_deactivated_components(search_space):
    c = []
    for sel in search_space:
        if not sel.attribute_value:
            c.append(sel)
    for sel in c:
        search_space.remove(sel)
    return search_space


def update_data(data, result):
    columns_to_delete = [key.attribute_name for key in result.results[0][1].selectors]
    indices_of_columns_to_delete = [data.activation_matrix.columns.get_loc(column) for column in columns_to_delete]
    data.probs[:, indices_of_columns_to_delete] = 0
    return data


def write_rules_to_the_file(rules, data, file_to_write, target):
    with open(file_to_write, "a"):
        for i in range(len(rules)):
            rule = list(rules[i])
            positive_support, negative_support = len(get_support_indices(rule, data.positive_data.to_numpy(),
                                                                         data.positive_graph_inds)), len(
                get_support_indices(
                    rule, data.negative_data.to_numpy(),
                    data.negative_graph_inds))
            rule.extend((positive_support, negative_support, positive_support + negative_support, target))
            rules[i] = rule
        df = pd.DataFrame(rules, columns=["SI", "Rule", "Positive_Support", "Negative_Support", "Total_Support", "target"])
        df.to_csv(file_to_write)


def beam_search(data, target, file_to_write, mode="single", max_rules=1):
    """
    :param file_to_write:
    :param data: a Data instance
    :param target: target class
    :param mode: either single to only take activated components into the account or double to have deactivated components
    :param max_rules: maximum number of rules to be mined
    :return:
    """
    rules = []
    search_space = ps.create_numeric_selectors(data.activation_matrix)
    if mode == "single":
        search_space = delete_deactivated_components(search_space)
    target = ps.BinaryTarget('label', bool(target))
    for i in tqdm(range(max_rules)):
        task = ps.SubgroupDiscoveryTask(data, target, search_space, result_set_size=1, depth=9,
                                        qf=SubjectiveInterestingness, min_quality=-math.inf)
        result = ps.BeamSearch().execute(task)
        data = update_data(data, result)
        data.split_probs()
        result = result.to_descriptions()
        # result.head(1).to_csv(file_to_write, mode="a", header=False if i else True)
        #print(result)
        rules.append(result[0])
    write_rules_to_the_file(rules, data, file_to_write, target)
