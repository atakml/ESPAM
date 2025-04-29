import torch

from evalutils import build_distance_matrix, build_ts
from loaders import pkl_loader, save_to_pkl
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def compute_KNN_for_protein(protein, weights, label_type, max_k="min_class" ,save_k=6, matrix_path=None, ts_path=None, smiles_path=None, class_zero_no_samples=None):
    KNN_acc = {}
    minor_class = 1 if label_type == "gt" else 0
    if smiles_path == None:
        smiles_path = f"data_OT_{protein}_{label_type}.pkl"
    smiles = pkl_loader(smiles_path)
    dist = build_distance_matrix(protein, label_type, weights, matrix_path, ts_path)
    labels = smiles["Responsive"].to_numpy()
    if max_k == "min_class":
        max_k = np.sum(labels==minor_class)
    if class_zero_no_samples == "min_class":
        class_zero_no_samples = max_k
    if class_zero_no_samples is not None:
        zero_indices = np.where(labels == minor_class)[0]
        try:
            selected_indices = np.random.choice(zero_indices, class_zero_no_samples, replace=False)
        except:
            print(zero_indices)
            print(class_zero_no_samples)
            print(labels)
            raise
        one_indices = np.where(labels == 1-minor_class)[0]
        new_indices = np.concatenate((selected_indices, one_indices))
        matrix_indices = np.ix_(new_indices, new_indices)
        dist = dist[matrix_indices]
        labels = labels[new_indices]

        
    for k in range(3, max_k):
        knn_distance_based1 = (
            KNeighborsClassifier(metric="precomputed", n_neighbors=k)
            .fit(dist, labels))  # smiles['class']))
        if (protein, weights) not in KNN_acc:
            KNN_acc[(protein, weights)] = [knn_distance_based1.score(dist, labels)]
        else:
            KNN_acc[(protein, weights)].append(knn_distance_based1.score(dist, labels))

        # print(knn_distance_based1.predict(dist))
        if k == save_k:
            save_to_pkl(knn_distance_based1.predict(dist), f"KNN_k={k}_{label_type}_{weights}_{protein}.pkl")
    return KNN_acc
