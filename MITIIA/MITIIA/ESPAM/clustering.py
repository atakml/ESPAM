import torch
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

from evalutils import build_distance_matrix, build_ts
from loaders import pkl_loader


def compute_KMedoids_for_protein(protein, weights, label_types, path=None):
    best_k = 2
    best_score = -1
    clusters = {}
    for k in range(3, 45):
        kmedoids = KMedoids(n_clusters=k, random_state=0)
        smiles = pkl_loader(f"data_OT_{protein}_{label_types}.pkl")
        dist = build_distance_matrix(protein, label_types, weights, path)
        clusters[(protein, weights, k)] = kmedoids.fit(dist).labels_
        score = silhouette_score(dist, clusters[(protein, weights, k)])
        if score > best_score:
            best_k, best_score = k, score
    clusters[(protein, weights)] = clusters[(protein, weights, best_k)]
    print(best_k)
    print("!!")
    return clusters[(protein, weights)]


def majority_score(cluster_labels, ground_labels):
    cluster_dict = {}
    for i in range(len(cluster_labels)):
        if cluster_labels[i] not in cluster_dict.keys():
            cluster_dict[cluster_labels[i]] = {}
        if ground_labels[i] not in cluster_dict[cluster_labels[i]].keys():
            cluster_dict[cluster_labels[i]][ground_labels[i]] = 1
        else:
            cluster_dict[cluster_labels[i]][ground_labels[i]] += 1
    final_labels = {}

    for i in range(len(cluster_labels)):
        cluster_label_count = cluster_dict[cluster_labels[i]]
        cluster_label_count[ground_labels[i]] -= 1
        #print(cluster_label_count)
        final_labels[i] = max(cluster_label_count.items(), key=lambda x: x[1])[0]
        cluster_label_count[ground_labels[i]] += 1
    return final_labels


def compute_clustering_score(clusters, protein, weights, label_types="model"):
    smiles = pkl_loader(f"data_OT_{protein}_{label_types}.pkl")
    labels = smiles["Responsive"].values
    cnt = 0
    tot = 0
    cluster_labels = majority_score(clusters, labels)
    for i in range(len(labels)):
        if labels[i] == cluster_labels[i]:
            cnt += 1
        tot += 1
    # score = rand_score(clusters[(protein, weights)], labels.values)
    score = cnt / tot
    return score, clusters