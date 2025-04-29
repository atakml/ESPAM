from ESPAM.mitihelper import compute_model_score, from_smiles_to_data


def fidelity(smiles, seq, selected_nodes, model, target_class, remove_mask_function):
    data = from_smiles_to_data(smiles)
    #z = torch.ones(data.x.shape[0]).bool()
    #z[center_indices] = False
    complement_mask = ~selected_nodes
    masked_graph = remove_mask_function(smiles, complement_mask)#features_to_graph(data, z, 1)
    return compute_model_score(data, seq, model, target_class) - compute_model_score(masked_graph, seq, model, target_class)


def infidelity(smiles, seq, selected_nodes, model, target_class, remove_mask_function):
    data = from_smiles_to_data(smiles)
    #z = torch.zeros(data.x.shape[0]).bool()
    #z[center_indices] = True
    complement_mask = selected_nodes
    masked_graph = remove_mask_function(smiles, complement_mask)#features_to_graph(data, z, 1)
    return compute_model_score(data, seq, model, target_class) - compute_model_score(masked_graph, seq, model, target_class)


def sparsity(smiles, selected_nodes, remove_mask_function):
    data = from_smiles_to_data(smiles)
    #z = torch.zeros(data.x.shape[0]).bool()
    #z[center_indices] = True
    #complement_mask = ~selected_nodes
    #print(complement_mask)

    masked_graph = remove_mask_function(smiles, selected_nodes)#features_to_graph(data, z, 1)
    return 1 - (masked_graph.true_edges)/(data.edge_index.shape[1])


def fidelity_acc(smiles, seq, selected_nodes, model, target_class, remove_mask_function):
    data = from_smiles_to_data(smiles)
    #z = torch.ones(data.x.shape[0]).bool()
    #z[center_indices] = False
    complement_mask = ~selected_nodes
    masked_graph = remove_mask_function(smiles, complement_mask)#features_to_graph(data, z, 1)
    return int((compute_model_score(data, seq, model, target_class)<0.5) == (compute_model_score(masked_graph, seq, model, target_class)<0.5))


def infidelity_acc(smiles, seq, selected_nodes, model, target_class, remove_mask_function):
    data = from_smiles_to_data(smiles)
    #z = torch.zeros(data.x.shape[0]).bool()
    #z[center_indices] = True
    complement_mask = selected_nodes
    masked_graph = remove_mask_function(smiles, complement_mask)#features_to_graph(data, z, 1)
    return int((compute_model_score(data, seq, model, target_class) <0.5) == (compute_model_score(masked_graph, seq, model, target_class)<0.5))