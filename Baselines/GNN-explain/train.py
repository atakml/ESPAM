import numpy as np

import torch
from tqdm import tqdm
import os
from torch_geometric.data import Batch, DataLoader as Dl
from model import SDNE
def eval_batch(model, data, target, opt=None, **kwargs) -> torch.tensor:
    if opt:
        opt.zero_grad()
    y_pred = model(data)
    loss = model.loss(y_pred, target)
    if opt:
        loss.backward()
        opt.step()
    if kwargs["verbose"]:
        print(float(loss))

    if type(y_pred)==tuple:
        return (y_pred[0].to(device="cpu"), y_pred[1].to(device="cpu"))
    return y_pred.detach().to(device="cpu")


def run_epoch(model, dataset, opt, device=torch.device('cpu'), **kwargs):
    preds = list()
    if opt:
        model.train()
    else:
        model.eval()
    targets = list()
    for data in dataset:  # tqdm(dataset,position=1):
        data.to(device)
        target = data.y if type(model) !=SDNE else data #if not it is a non supervised task
        evaluation = eval_batch(
            model,
            data,
            target,
            opt,
            metric_funs=kwargs["metrics"],
            **kwargs
        )
        preds.append(evaluation)
        targets.append(target.to(device="cpu"))

    if type(model) != SDNE:
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        return [metr(preds, targets) for metr in kwargs["metrics"]]
    else:
        preds = tuple((torch.cat([el[i] for el in preds]) for i in [0, 1]))
        targets = [el for el in Dl(dataset.dataset, batch_size=len(dataset.dataset),shuffle=False)][0]
        return [metr(preds, targets).detach() for metr in kwargs["metrics"]]


def get_gradient(model, dataset, device=torch.device('cpu'), **kwargs):
    grads_x = list()
    grads_e = list()

    outputs = list()
    for data in dataset:
        model.zero_grad()
        data.to(device)
        data.x.requires_grad = True
        # data.edge_index.requires_grad=True

        output = model.forward(data)
        # "= torch.argmax(output, 1).item()
        output[1].backward(retain_graph=True)  # 1 parce que c'est la précence
        gradient_x = data.x.grad.detach().cpu()
        # gradient_e = data.edge_index.grad.detach().cpu()

        grads_x.append(gradient_x)
        # grads_e.append(gradient_e)
        outputs.append(output[1])
    return grads_x, grads_e, outputs


def get_pf_str(train, valid, **kwargs):
    ret = ""
    if train[0]:
        ret += "train loss: {:.3f}".format(float(train[0]))
        if kwargs.get("print_all"):
            for el in train[1:]:
                ret += ", {:.3f}".format(float(el))
    if valid[0]:
        ret += "| valid loss: {:.3f}".format(float(valid[0]))
        if kwargs.get("print_all"):
            for el in valid[1:]:
                ret += ", {:.3f}".format(float(el))
    return ret


from utils.dynamic_plot import update_line


def fit(model, opt, trainset=None, testset=None, epochs=1, **kwargs):
    out = list()
    bar = tqdm(range(epochs))
    for epoch in bar:
        kwargs["lam"] = adaptation_factor(epoch * 1.0 / epochs)

        # Equals None if dataset is none, else the value of run_epoch
        train_metrics = trainset and run_epoch(model, trainset, opt, **kwargs)
        valid_metrics = testset and run_epoch(model, testset, None, **kwargs)
        if len(train_metrics):
            bar.set_postfix_str(get_pf_str(train_metrics, valid_metrics, **kwargs))

            # bar.set_postfix({"train_loss":float(train_loss), "test_loss":float(valid_loss)})
        out.append((train_metrics, valid_metrics))
        # update_line(train_metrics[0], valid_metrics[0])
        if kwargs.get("save_step"):
            file = open(os.path.join(kwargs["save_dir"], kwargs["save"] + "_loss"), "wb")
            np.save(file, out)
            torch.save(model.state_dict(), os.path.join(kwargs["save_dir"], kwargs["save"] + 'step'))

    return out

import dataloader
def fit_autoencoder(model, opt, trainset=None, testset=None, epochs=1, **kwargs):
    out = list()
    bar = tqdm(range(epochs))
    tr = dataloader.autoencoder_transform()
    train_tr = trainset.dataset.dataset.transform
    trainset.dataset.dataset.transform=tr
    test_tr =testset.dataset.dataset.transform
    testset.dataset.dataset.transform=tr


    for epoch in bar:
        kwargs["lam"] = adaptation_factor(epoch * 1.0 / epochs)

        # Equals None if dataset is none, else the value of run_epoch
        train_metrics = trainset and run_epoch(model, trainset, opt, **kwargs)
        valid_metrics = testset and run_epoch(model, testset, None, **kwargs)
        if len(train_metrics):
            bar.set_postfix_str(get_pf_str(train_metrics, valid_metrics, **kwargs))

            # bar.set_postfix({"train_loss":float(train_loss), "test_loss":float(valid_loss)})
        out.append((train_metrics, valid_metrics))
        # update_line(train_metrics[0], valid_metrics[0])
        if kwargs.get("save_step"):
            file = open(os.path.join(kwargs["save_dir"], kwargs["save"] + "_loss"), "wb")
            np.save(file, out)
            torch.save(model.state_dict(), os.path.join(kwargs["save_dir"], kwargs["save"] + 'step'))
    trainset.dataset.dataset.transform = train_tr
    testset.dataset.dataset.transform = test_tr
    return out




# utils
def adaptation_factor(epoch):
    return 2 / (1 + np.exp(- 10 * epoch)) - 1


def one_hot(batch, classes):
    ones = torch.eye(classes)
    return ones.index_select(0, batch)


