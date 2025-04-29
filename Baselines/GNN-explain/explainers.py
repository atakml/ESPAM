import os
import torch.nn as nn

import torch
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader

import train


def integrated_gradient(model:nn.Module, data:Data, baseline:Data, steps:int, **kwargs):
    # x, edge_index, batch = data.x, data.edge_index, data.batch
    # bx, bedge_index, bbatch = baselines.x, baselines.edge_index, baselines.batch

    x, edge_index = data.x, data.edge_index
    bx, bedge_index = baseline.x, baseline.edge_index

    scaled_x = [bx + (float(i) / steps) * (x - bx) for i in range(0, steps + 1)]
    scaled_edge_index = [bedge_index + (float(i) / steps) * (edge_index - bedge_index) for i in range(0, steps + 1)]

    scaled_dset = [Data(x=x, edge_index=edge_index, y=0) for x, edge, in zip(scaled_x, scaled_edge_index)]
    scaled = DataLoader(scaled_dset, batch_size=1, shuffle=False)

    grads_x, grads_e, outs = train.get_gradient(model, scaled, **kwargs)
    grads_x = torch.stack(grads_x)
    # grads_e = torch.stack(grads_e)

    # grads, _ = predict_and_gradients(scaled_inputs, model, target_label_idx, cuda)
    avg_grads_x = torch.mean(grads_x, dim=0)
    # avg_grads_e = torch.mean(grads_e, axis=0)

    integrated_grad_x = (x - bx) * avg_grads_x
    # integrated_grad_e = (edge_index - bedge_index) * avg_grads_e

    # print(integrated_grad_x.sum().item(), (outs[-1]-outs[0]).item())
    # print(integrated_grad_e.sum().item(), (outs[-1]-outs[0]).item())

    return integrated_grad_x  # , integrated_grad_e


def use_integrated_gradient(model:nn.Module, dataset:DataLoader, **kwargs):
    newdl = DataLoader(dataset.dataset, batch_size=1, shuffle=False)
    grad = list()
    for el in tqdm(newdl):
        x, edge_index = el.x, el.edge_index
        example = Data(x=x, edge_index=edge_index)
        baseline = Data(x=torch.zeros(x.shape), edge_index=torch.zeros(edge_index.shape), y=0)

        grad.append(integrated_gradient(model, example, baseline, 100, **kwargs))
    return grad


def save_integrated_gradient(model:nn.Module, dataset_name:str, dirname:str, reset_storage=False, **kwargs):
    dataset = kwargs[dataset_name]

    if reset_storage or not os.path.exists(dirname):
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        expls = use_integrated_gradient(model, dataset, **kwargs)
        data_list = list(dataset.dataset)
        torch.save(data_list, os.path.join(dirname, "data_list.pth"))
        tmp = [(e, i) for i, e in enumerate(expls)]
        torch.save({"integrated_gradient": tmp}, os.path.join(dirname, "gradients.pth"))


def select_nodes(model:nn.Module, element:Data, k:int, **kwargs):
    x, edge_index = element.x, element.edge_index
    baseline = Data(x=torch.zeros(x.shape), edge_index=torch.zeros(edge_index.shape), y=0)
    expl = integrated_gradient(model, element, baseline, 100, **kwargs)
    els = [(el, i) for i, el in enumerate(expl)]
    els.sort(key=lambda el: el[0].sum(), reverse=True)
    return [el[1] for el in els[:k]]


def select_nodes_from_data(igrad, el_index, k, **kwargs):
    expl = igrad[el_index]
    els = [(el, i) for i, el in enumerate(expl[0])]
    els.sort(key=lambda x: x[0].sum(), reverse=True)
    return [el[1] for el in els[:k]]


def shap(model, example, **kwargs):
    pass
