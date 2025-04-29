import os

import torch
from tqdm import tqdm

import clustering
import dataloader
from model import GCNet,SDNE
from train import fit, fit_autoencoder
from utils.args import *
from collections import defaultdict
from utils.plot_learning_curves import plot_learning_curve

from torch_geometric.data import DataLoader as Dl

import matplotlib.pyplot as plt

import numpy as np

from utils import plot_molecule
import distance
from distance import word_moover_distance_experiment,graph_edit_distance, gnn_ged_experiment
from distance import gmd_AE_distance_experiment,ged_AE_distance_experiment
import graph_selection

def base_test(**kwargs):

    kwargs.update(base_args())
    #kwargs.update(use_smiles(kwargs))
    kwargs.update(use_bbbp(kwargs))
    #kwargs.update(use_smiles_edonicity(kwargs))
    #kwargs.update(use_AIDS(kwargs))
    kwargs["reset_storage"] = True

    model = get_model(**kwargs)



def dataset_balance(**kwargs):
    kwargs.update(base_args())
    kwargs.update(use_smiles(kwargs))
    #kwargs.update(use_bbbp(kwargs))
    #.kwargs.update(use_AIDS(kwargs))

    train = kwargs["trainset"]
    test = kwargs["testset"]
    train_balance = train.dataset.dataset.data.y.to(float).mean()
    test_balance = test.dataset.dataset.data.y.to(float).mean()

    print("train balance", train_balance)
    print("test balance", test_balance)

def get_model(trainset, testset, input_dim, hidden_dims, decoder_dims, output_dim, loss, lr, weight_decay,
              betas, reset_storage=False, **kwargs):
    file = os.path.join(kwargs["model_dir"], kwargs["name"]+".pth")
    model = GCNet(
        input_dim,
        hidden_dims,
        decoder_dims,
        output_dim,
        loss).to(kwargs["device"])

    if reset_storage or not os.path.exists(file):
        if not os.path.exists(kwargs["model_dir"]):
            os.mkdir(kwargs["model_dir"])
        optim = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )
        metrics = fit(model, optim, trainset, testset, **kwargs)
        model.to("cpu")
        torch.save(model.state_dict(), file)

    model.load_state_dict(torch.load(file))
    model.to(kwargs["device"])
    if kwargs["get_metrics"]:
        return model, metrics
    return model

def get_model_ae(trainset, testset, input_dim, encoder_dims, embedding_dim, decoder_dims, loss, lr, weight_decay,
              betas, reset_storage=False, **kwargs):
    file = os.path.join(kwargs["model_dir"], kwargs["name"]+".pth")
    model = SDNE(
        input_dim,
        encoder_dims,
        embedding_dim,
        decoder_dims).to(kwargs["device"])

    if reset_storage or not os.path.exists(file):
        if not os.path.exists(kwargs["model_dir"]):
            os.mkdir(kwargs["model_dir"])
        optim = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )
        metrics = fit_autoencoder(model, optim, trainset, testset, **kwargs)
        model.to("cpu")
        torch.save(model.state_dict(), file)

    model.load_state_dict(torch.load(file))
    model.to(kwargs["device"])
    if kwargs["get_metrics"]:
        return model, metrics
    return model


def get_trace(model, dataset_name, reset_storage=False, **kwargs):
    dirname = os.path.join("datasets","trace", kwargs["name"])

    file_names = [os.path.join(dirname, dataset_name+"_layer_" + str(i) + ".pth") for i in range(len(model.layers))]
    if reset_storage or any([not os.path.exists(f) for f in file_names]):
        if not os.path.exists('datasets/trace'):
            os.mkdir("datasets/trace")
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        data, data_list = clustering.extract_data(model, kwargs[dataset_name], **kwargs)
        torch.save(data_list, os.path.join(dirname, dataset_name + "_data_list.pth"))
        for i in range(len(model.layers)):
            tmp = [((el[0][0].cpu().detach(),el[0][1].cpu().detach()), el[1]) for el in data if el[2] == i]
            #((forward, backward),index)
            torch.save(tmp, file_names[i])
    return [torch.load(f) for f in file_names]#, torch.load(os.path.join(dirname, dataset_name + "_data_list.pth"))

def get_ae_embedding(model, dataset_name, reset_storage=False, **kwargs):
    dirname = os.path.join("datasets", "trace", kwargs["name"])

    file_name = os.path.join(dirname, dataset_name+ ".pth")
    if reset_storage or not os.path.exists(file_name):
        if not os.path.exists('datasets/trace'):
            os.mkdir("datasets/trace")
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        data, data_list = clustering.compute_ae_embedding(model, kwargs[dataset_name], **kwargs)
        torch.save(data_list, os.path.join(dirname, dataset_name + "_data_list.pth"))
        tmp = [((el[0].cpu().detach(),0), el[1]) for el in data]
        torch.save(tmp, file_name)

    return torch.load(file_name) #, torch.load(os.path.join(dirname, dataset_name + "_data_list.pth"))




import random

def retrain_experiment(**kwargs):
    kwargs.update(base_args())
    kwargs.update(use_smiles(kwargs))
    #kwargs.update(use_bbbp(kwargs))
    #kwargs.update(use_AIDS(kwargs))
    kwargs["get_metrics"] = True
    kwargs["reset_storage"] = True
    kwargs["epochs"] = 80
    kwargs["save_step"]= False
    # grid search
    plt.rc('figure', max_open_warning = 120)

    _, axes = plt.subplots(10,5, figsize=(60,60))
    res = [[0] * 5 for _ in range(10)]

    nlayers = [1,2,3,4,5,6,8]
    layer_size = [10,20,35,55]
    res = [[0] * len(layer_size) for _ in range(len(nlayers))]
    _, axes = plt.subplots(len(nlayers),len(layer_size), figsize=(60,60))
    for i, vi in enumerate(nlayers):# n-layers
        for j, vj in enumerate(layer_size):# layer size
            kwargs["hidden_dims"] = [vj] * vi
            datas = list()
            for k in range(10):
                init_fn = lambda x: np.random.seed(random.randint(0,2**53-1))
                kwargs["trainset"]= Dl(kwargs["trainset"].dataset,batch_size=kwargs["batch_size"], shuffle=True,worker_init_fn=init_fn)
                model, data = get_model(**kwargs)
                datas.append(data)
            values = plot_learning_curve(datas,axes=axes,i=i,j=j, indexes=[-2])
            res[i][j] = values
            #axes[i, j]= plot
    plt.show()
    caption = kwargs["name"]+ " "
    metrics = ["train loss","test loss", "train AU ROC", "test AU ROC"]
    print(print_res_to_latex(res, nlayers, layer_size, caption, metrics))

def print_res_to_latex(res, layers, layer_size, caption, metrics):
    size = len(res[0][0])
    out = ""
    for i in range(size):
        out += "\\begin{figure}\n "
        out += "\\begin{tabular}{|"
        for _ in range(len(layer_size)+1):
            out += "c|"
        out += "}\\hline\n"
        for s in layer_size:
            out += " & " + str(s)
        out += "\\\\ \hline \n"
        for j, l in enumerate(res):
            out += str(layers[j])
            for elm in l:
                el = elm[i]
                out += "& $" + str("{:0.3}".format(float(el[0][-1])))
                out += "\\pm" + str("{:0.3}".format(float(el[1][-1]))) +" $ "
            out+="\\\\ \\hline\n"
        out +="\\end{tabular}\n\n"
        out += "\\caption{" +  caption + metrics[i] + "}\n"
        out += "\\end{figure}\n\n "
    return out



def test_explain(**kwargs):
    kwargs.update(base_args())
    kwargs.update(use_smiles(kwargs))
    # kwargs.update(use_bbbp(kwargs))
    kwargs["reset_storage"] = False

    model = get_model(**kwargs)

    dataset_file = os.path.join("datasets", kwargs["new_data_dir"], kwargs["new_data"])
    datas, data_list = get_trace(model, "trainset", train, **kwargs)

    datas = list(map(clustering.molecular_maping, datas))

    data = {k: [datas[i][k] for i in range(len(datas))] for k in datas[0].keys()}
    values = [[l[0][1] for l in lay] for lay in v]
    # norm = lambda x: x.norm(2)
    # norm = lambda x: x.norm(10)
    norm = lambda x: x.max()
    values = [torch.tensor([norm(el) for el in lay]) for lay in values]
    values = [(lay / max(lay)).tolist() for lay in values]

    to_plot = [
        (data_list[k], lay) for lay in values]
    plot_molecule.plot_sal(*to_plot)



def autoencoder_eperiment(**kwargs):
    kwargs.update(base_args())
    kwargs.update(use_smiles(kwargs))
    # kwargs.update(use_bbbp(kwargs))
    # kwargs.update(use_AIDS(kwargs))
    kwargs.update(use_SDNE(kwargs))
    kwargs["reset_storage"] = True
    kwargs["name"]+="_AE"
    model = get_model_ae(**kwargs)




def base_rule_system(**kwargs):
    kwargs.update(base_args())
    kwargs.update(use_smiles(kwargs))
    #kwargs.update(use_bbbp(kwargs))
    #kwargs.update(use_AIDS(kwargs))

    kwargs["reset_storage"] = False

    #model.load_state_dict(torch.load(model_file))

    model = get_model(**kwargs)
    dataset_file = os.path.join("datasets", kwargs["new_data_dir"], kwargs["new_data"])
    data, data_list= get_trace(model, "trainset", "train", **kwargs)
    #explainers.save_integrated_gradient(model, "trainset", dataset_file, **kwargs)

    #train_igrad = torch.load(os.path.join(dataset_file, "gradients.pth"))["integrated_gradient"]

    metric_distance_experiment(data, data_list, **kwargs)

    embedding_dict = defaultdict(list)
    for i, layer in enumerate(data):
        for l in layer:
            embedding_dict[(tuple(l[0][0].tolist()),i)].append(float(data_list[l[1]].y))
            #maps

    mean= float(torch.tensor([float(el.y) for el in kwargs["trainset"].dataset for _ in range(el.x.size(0))]).mean())
    vals = [(sum(el),len(el)-sum(el),i) for (_,i), el in embedding_dict.items()]
    #vals = [(sum(el)/len(el), len(el),i ) for (_,i), el in embedding_dict.items()]
    #vals = list(sorted(filter(lambda x:x[1]>100,vals), key=lambda x: abs(x[0])))
    #vals = list(sorted(filter(lambda x:1000>x[0] ,vals), key=lambda x: abs(x[0])))

    plot_molecule.plot_rules(vals,mean)

    #print(vals)


def graph_selection_experiment(**kwargs):
    kwargs.update(base_args())
    #kwargs.update(use_smiles(kwargs))
    #kwargs.update(use_bbbp(kwargs))
    kwargs.update(use_AIDS(kwargs))

    kwargs["reset_storage"] = False


    model = get_model(**kwargs)
    dataset_file = os.path.join("datasets", kwargs["new_data_dir"], kwargs["new_data"])
    train_data, train_data_list = get_trace(model, "trainset", "train", **kwargs)
    test_data, test_data_list = get_trace(model, "testset", "test", **kwargs)

    # explainers.save_integrated_gradient(model, "trainset", dataset_file, **kwargs)


    # train_igrad = torch.load(os.path.join(dataset_file, "gradients.pth"))["integrated_gradient"]
    layer = 5

    train_graph_list = distance.get_graph_list(train_data, layer)
    test_graph_list = distance.get_graph_list(test_data, layer)

    distances = distance.get_graph_distances(
        test_graph_list=test_graph_list,
        data=train_data,
        graph_list=train_graph_list,
        layer=layer,
        file_tag="test_tr", **kwargs)
    distances = word_moover_distance_experiment(test_data, test_data_list, train_data, train_data_list,file_tag="test_tr", **kwargs)
    distances2 = distance.get_graph_distances(
        test_graph_list=train_graph_list,
        data=train_data,
        graph_list=train_graph_list,
        layer=layer,
        file_tag="train_tr", **kwargs)
    #distances2 = word_moover_distance_experiment(train_data, train_data_list, train_data, train_data_list, file_tag="train_tr", **kwargs)


    train_graphs = clustering.molecular_maping(train_data[layer])

    train_graph_list = [0 for _ in range(len(train_graphs))]
    for i, el in train_graphs.items():
        train_graph_list[i] = el
    classes = [train_data_list[g[0][1]].y for g in train_graph_list]

    ratio = [20,20]
    for i in range(len(distances)):
        ret = graph_selection.select_graphs(distances[i], distances2,classes, ratio)
        print(ret)


        ids = np.where(ret)[0]
        mols =[test_data_list[i]]
        mols += [train_data_list[i] for i in ids]
        plot_molecule.plot_mol(size=[2,21], molecules=mols)




def kmean_search(**kwargs):
    kwargs.update(base_args())
    kwargs["layer"] = 5
    dataset_file = os.path.join("datasets", kwargs["new_data_dir"], kwargs["new_data"])
    data_list = torch.load(os.path.join(dataset_file, "data_list.pth"))
    data = torch.load(os.path.join(dataset_file, "layer_" + str(kwargs["layer"]) + ".pth"))["feature_data"]
    # data= clustering.molecular_maping(data)

    clustering.clusters(data, data_list, **kwargs)



def metrics_experiment(**kwargs):
    kwargs.update(base_args())
    kwargs.update(use_smiles(kwargs))
    #kwargs.update(use_bbbp(kwargs))

    kwargs["reset_storage"] = False

    model = get_model(**kwargs)

    dataset_file = os.path.join("datasets", kwargs["new_data_dir"], kwargs["new_data"])
    get_trace(model, "trainset", "train", **kwargs)
    #explainers.save_integrated_gradient(model, "trainset", dataset_file, **kwargs)

    data_list = torch.load(os.path.join(dataset_file, "data_list.pth"))
    data = [torch.load(os.path.join(dataset_file, "layer_" + str(l) + ".pth"))["feature_data"] for l in range(len(kwargs["hidden_dims"]))]
    #train_igrad = torch.load(os.path.join(dataset_file, "gradients.pth"))["integrated_gradient"]

def gmd_exp(**kwargs):
    kwargs.update(base_args())
    kwargs["device"] =torch.device('cpu')

    #kwargs.update(use_smiles(kwargs))
    #kwargs.update(use_bbbp(kwargs))
    kwargs.update(use_AIDS(kwargs))

    kwargs["reset_storage"] = False
    kwargs["knn_metrics"] = ["acc", "aur", "kl"]

    model = get_model(**kwargs)

    kwargs["trainset"] = Dl(kwargs["trainset"].dataset, batch_size=1, shuffle=False)
    kwargs["testset"] = Dl(kwargs["testset"].dataset, batch_size=1, shuffle=False)
    train_data = get_trace(model, "trainset", **kwargs)
    test_data = get_trace(model, "testset", **kwargs)

    caption = kwargs["name"] + " train"
    distances2 = word_moover_distance_experiment(train_data, train_data, file_tag="trainset", caption=caption, **kwargs)
    caption = kwargs["name"] + " test"
    distances = word_moover_distance_experiment(train_data, test_data,file_tag="testset", caption=caption, **kwargs)

    kwargs["reset_storage"] = False
    kwargs["threshold"] = get_threshold(model, kwargs["trainset"])
    kwargs["trainset"] = dataloader.transform_dataset_labels(kwargs["trainset"], lambda x: model(x).detach().numpy()[1])
    kwargs["testset"] = dataloader.transform_dataset_labels(kwargs["testset"], lambda x: model(x).detach().numpy()[1])

    caption = kwargs["name"] + " train fidelity"
    distances3 = word_moover_distance_experiment(train_data, train_data,file_tag="trainset", caption=caption, **kwargs)

    caption = kwargs["name"] + " test fidelity"
    distances4 = word_moover_distance_experiment(train_data, test_data,file_tag="testset", caption=caption,  **kwargs)




def ged_experiment(**kwargs):
    kwargs.update(base_args())
    kwargs["device"] =torch.device('cpu')

    #kwargs.update(use_smiles(kwargs))
    #kwargs.update(use_bbbp(kwargs))
    kwargs.update(use_AIDS(kwargs))
    kwargs["name"] += "_ged"
    kwargs["reset_storage"] = False
    kwargs["knn_metrics"] = ["acc", "aur", "kl"]

    model = get_model(**kwargs)
    kwargs["trainset"] = Dl(kwargs["trainset"].dataset, batch_size=1, shuffle=False)
    kwargs["testset"] = Dl(kwargs["testset"].dataset, batch_size=1, shuffle=False)

    caption = kwargs["name"] + " train"
    v1 = graph_edit_distance(file_tag="trainset",caption=caption, **kwargs)
    caption = kwargs["name"] + " test"
    v2 = graph_edit_distance(file_tag="testset",caption=caption, **kwargs)

    kwargs["reset_storage"] = False
    kwargs["threshold"] = get_threshold(model, kwargs["trainset"])
    kwargs["trainset"] = dataloader.transform_dataset_labels(kwargs["trainset"], lambda x: model(x).detach().numpy()[1])
    kwargs["testset"] = dataloader.transform_dataset_labels(kwargs["testset"], lambda x: model(x).detach().numpy()[1])

    caption = kwargs["name"] + " train fidelity"
    v3 = graph_edit_distance(file_tag="trainset",caption=caption, **kwargs)
    caption = kwargs["name"] + " test fidelity"
    v4 = graph_edit_distance(file_tag="testset",caption=caption, **kwargs)

    print(distance.print_res_to_latex([v1,v2,v3,v4],
                                      ["train","test", "train fidelity" ,"test fidelity"],
                                      [2 ** k for k in range(8)],
                                      "",
                                      kwargs["knn_metrics"]))


def get_model_output(model, dataloader:Dl):
    dset = Dl(dataloader.dataset, batch_size=1, shuffle=False)
    return np.array([model(el).tolist()[1] for el in tqdm(dset)])


import metrics
def get_threshold(model, dataloader:Dl):
    dl=  Dl(dataloader.dataset, batch_size=1, shuffle=False)
    model.eval()
    target = [el.y.numpy() for el in dl]
    pred = [model(el).detach().numpy()[1] for el in dl]
    return metrics.optimal_threshold(pred, target)
    #model_output = get_model_output(model, dl)
    #train_labels = np.array([el.y.numpy() for el in dl]).astype(bool)
    #return np.argmax([((np.array(model_output)>(th/1000)).astype(bool)==train_labels).mean()
    #                  for th in range(1000)])/1000



def gnn_ged(**kwargs):
    kwargs.update(base_args())
    kwargs["device"] =torch.device('cpu')

    kwargs.update(use_smiles(kwargs))
    #kwargs.update(use_bbbp(kwargs))
    #kwargs.update(use_AIDS(kwargs))

    kwargs["reset_storage"] = False
    kwargs["knn_metrics"] = ["acc", "aur", "kl"]
    model = get_model(**kwargs)

    train_data = get_trace(model, "trainset", **kwargs)
    test_data = get_trace(model, "testset", **kwargs)

    kwargs["trainset"] = Dl(kwargs["trainset"].dataset, batch_size=1, shuffle=False)
    kwargs["testset"] = Dl(kwargs["testset"].dataset, batch_size=1, shuffle=False)

    caption = kwargs["name"] + " train"
    v = gnn_ged_experiment(train_data, train_data, file_tag="trainset", caption=caption, **kwargs)
    caption = kwargs["name"] + " test"
    v = gnn_ged_experiment(train_data, test_data, file_tag="testset", caption=caption, **kwargs)

    kwargs["reset_storage"] = False
    kwargs["threshold"] = get_threshold(model, kwargs["trainset"])
    kwargs["trainset"] = dataloader.transform_dataset_labels(kwargs["trainset"], lambda x: model(x).detach().numpy()[1])
    kwargs["testset"] = dataloader.transform_dataset_labels(kwargs["testset"], lambda x: model(x).detach().numpy()[1])

    caption = kwargs["name"] + " train fidelity"
    v = gnn_ged_experiment(train_data, train_data, file_tag="trainset", caption=caption, **kwargs)

    caption = kwargs["name"] + " test fidelity"
    v = gnn_ged_experiment(train_data, test_data, file_tag="testset", caption=caption, **kwargs)


from distance import  jaccard_distance_experiment

def jaccard_experiment(**kwargs):
    kwargs.update(base_args())
    kwargs["device"] =torch.device('cpu')

    kwargs.update(use_smiles(kwargs))
    #kwargs.update(use_bbbp(kwargs))
    #kwargs.update(use_AIDS(kwargs))

    kwargs["knn_metrics"] = ["acc", "aur", "kl"]

    kwargs["reset_storage"] = False
    model = get_model(**kwargs)
    kwargs["trainset"] = Dl(kwargs["trainset"].dataset, batch_size=1, shuffle=False)
    kwargs["testset"] = Dl(kwargs["testset"].dataset, batch_size=1, shuffle=False)

    caption = kwargs["name"] + " train"
    v1 = jaccard_distance_experiment(file_tag="trainset", caption=caption, **kwargs)
    caption = kwargs["name"] + " test"
    v2 = jaccard_distance_experiment(file_tag="testset", caption=caption, **kwargs)

    kwargs["reset_storage"] = False
    kwargs["threshold"] = get_threshold(model, kwargs["trainset"])

    kwargs["trainset"] = dataloader.transform_dataset_labels(kwargs["trainset"], lambda x: model(x).detach().numpy()[1])
    kwargs["testset"] = dataloader.transform_dataset_labels(kwargs["testset"], lambda x: model(x).detach().numpy()[1])

    caption = kwargs["name"] + " train fidelity"
    v3 = jaccard_distance_experiment(file_tag="trainset", caption=caption, **kwargs)

    caption = kwargs["name"] + " test fidelity"
    v4 = jaccard_distance_experiment(file_tag="testset", caption=caption,  **kwargs)

    print(distance.print_res_to_latex([v1,v2,v3,v4],
                                      ["train","test", "train fidelity" ,"test fidelity"],
                                      [2 ** k for k in range(8)],
                                      "",
                                      kwargs["knn_metrics"]))



def gmd_AE(**kwargs):
    kwargs.update(base_args())
    #kwargs.update(use_smiles(kwargs))
    #kwargs.update(use_bbbp(kwargs))
    kwargs.update(use_AIDS(kwargs))
    base_model = get_model(**kwargs).to("cpu")
    kwargs.update(use_SDNE(kwargs))
    kwargs["reset_storage"] = False


    kwargs["name"]+="gmd_AE_"
    model = get_model_ae(**kwargs)
    kwargs["knn_metrics"] = ["acc", "aur", "kl"]


    kwargs["trainset"] = Dl(kwargs["trainset"].dataset, batch_size=1, shuffle=False)
    kwargs["testset"] = Dl(kwargs["testset"].dataset, batch_size=1, shuffle=False)
    train_data = get_ae_embedding(model, "trainset", **kwargs)
    test_data = get_ae_embedding(model, "testset", **kwargs)

    caption = kwargs["name"] + " train"
    v1 = gmd_AE_distance_experiment(train_data, train_data, file_tag="trainset", caption=caption, **kwargs)
    caption = kwargs["name"] + " test"
    v2 = gmd_AE_distance_experiment(train_data, test_data,file_tag="testset", caption=caption, **kwargs)

    kwargs["reset_storage"] = False

    kwargs["threshold"] = get_threshold(base_model, kwargs["trainset"])
    kwargs["trainset"] = dataloader.transform_dataset_labels(kwargs["trainset"], lambda x: base_model(x).detach().numpy()[1])
    kwargs["testset"] = dataloader.transform_dataset_labels(kwargs["testset"], lambda x: base_model(x).detach().numpy()[1])

    caption = kwargs["name"] + " train fidelity"
    v3 = gmd_AE_distance_experiment(train_data, train_data,file_tag="trainset", caption=caption, **kwargs)

    caption = kwargs["name"] + " test fidelity"
    v4 = gmd_AE_distance_experiment(train_data, test_data,file_tag="testset", caption=caption,  **kwargs)


    print(distance.print_res_to_latex([v1,v2,v3,v4],
                                      ["train","test", "train fidelity" ,"test fidelity"],
                                      [2 ** k for k in range(8)],
                                      "",
                                      kwargs["knn_metrics"]))



def ged_AE(**kwargs):
    kwargs.update(base_args())
    #kwargs.update(use_smiles(kwargs))
    kwargs.update(use_bbbp(kwargs))
    #kwargs.update(use_AIDS(kwargs))
    base_model = get_model(**kwargs).to("cpu")
    kwargs.update(use_SDNE(kwargs))
    kwargs["reset_storage"] = False


    kwargs["name"]+="_ged_AE_"
    model = get_model_ae(**kwargs)
    kwargs["knn_metrics"] = ["acc", "aur", "kl"]


    kwargs["trainset"] = Dl(kwargs["trainset"].dataset, batch_size=1, shuffle=False)
    kwargs["testset"] = Dl(kwargs["testset"].dataset, batch_size=1, shuffle=False)
    train_data = get_ae_embedding(model, "trainset", **kwargs)
    test_data = get_ae_embedding(model, "testset", **kwargs)

    caption = kwargs["name"] + " train"
    v1 = ged_AE_distance_experiment(train_data, train_data, file_tag="trainset", caption=caption, **kwargs)
    caption = kwargs["name"] + " test"
    v2 = ged_AE_distance_experiment(train_data, test_data,file_tag="testset", caption=caption, **kwargs)

    kwargs["reset_storage"] = False

    kwargs["threshold"] = get_threshold(base_model, kwargs["trainset"])
    kwargs["trainset"] = dataloader.transform_dataset_labels(kwargs["trainset"], lambda x: base_model(x).detach().numpy()[1])
    kwargs["testset"] = dataloader.transform_dataset_labels(kwargs["testset"], lambda x: base_model(x).detach().numpy()[1])

    caption = kwargs["name"] + " train fidelity"
    v3 = ged_AE_distance_experiment(train_data, train_data,file_tag="trainset", caption=caption, **kwargs)

    caption = kwargs["name"] + " test fidelity"
    v4 = ged_AE_distance_experiment(train_data, test_data,file_tag="testset", caption=caption,  **kwargs)


    print(distance.print_res_to_latex([v1,v2,v3,v4],
                                      ["train","test", "train fidelity" ,"test fidelity"],
                                      [2 ** k for k in range(8)],
                                      kwargs["name"],
                                      kwargs["knn_metrics"]))


import WL
def wl_xperiment(**kwargs):
    kwargs.update(base_args())
    kwargs.update(use_smiles(kwargs))
    #kwargs.update(use_bbbp(kwargs))
    #kwargs.update(use_AIDS(kwargs))
    base_model = get_model(**kwargs).to("cpu")
    kwargs["reset_storage"] = False

    train_graph = distance.dataset_to_nx(kwargs["trainset"])
    train_labels = [el.y.numpy()[0] for el in kwargs["trainset"].dataset]
    test_graph = distance.dataset_to_nx(kwargs["testset"])
    test_labels = [el.y.numpy()[0] for el in kwargs["testset"].dataset]

    km = dict()
    train_features = list()
    for g in tqdm(train_graph):
        mset, km = WL.wl(g, km,5)
        train_features.append(mset)
    model = WL.wl_kernel(train_features, train_labels, km)

    WL.test_kernel(model,train_features,train_labels )
    test_features = list()
    for g in tqdm(test_graph):
        mset, km = WL.wl(g, km,5)
        test_features.append(mset)

    WL.test_kernel(model,test_features,test_labels )


from utils import tb
import numpy as np
def plot_dataset(**kwargs):
    kwargs.update(base_args())
    #kwargs["device"] =torch.device('cpu')

    #kwargs.update(use_smiles(kwargs))
    #kwargs.update(use_bbbp(kwargs))
    kwargs.update(use_AIDS(kwargs))

    layer = len(kwargs["hidden_dims"])-1
    #tb.write_dset("trainset", **kwargs)

    #l = len(kwargs["trainset"].dataset)
    #distance_matrix = np.random.random((l,l))
    for dataset in "trainset","testset" :

        distance_matrix = distance.get_distance_matrix(file_tag=dataset,medhod_name="ged", **kwargs)
        tb.nearest_tag("ged ", dataset, distance_matrix, **kwargs)
        distance_matrix = distance.get_distance_matrix(file_tag=dataset,medhod_name="jaccard", **kwargs)
        tb.nearest_tag("jaccard", dataset, distance_matrix, **kwargs)
        distance_matrix = distance.get_distance_matrix(file_tag=str(layer)+"_"+dataset,medhod_name="gmd", **kwargs)
        tb.nearest_tag("gmd", dataset, distance_matrix, **kwargs)
        distance_matrix = distance.get_distance_matrix(file_tag=str(layer) + "_"+dataset,medhod_name="hungarian", **kwargs)
        tb.nearest_tag("hungarian", dataset, distance_matrix, **kwargs)

def plot_cmc(**kwargs):
    kwargs.update(base_args())
    #kwargs["device"] =torch.device('cpu')

    #kwargs.update(use_smiles(kwargs))
    #kwargs.update(use_bbbp(kwargs))
    kwargs.update(use_AIDS(kwargs))

    layer = len(kwargs["hidden_dims"])-1
    #tb.write_dset("trainset", **kwargs)

    #l = len(kwargs["trainset"].dataset)
    #distance_matrix = np.random.random((l,l))
    #tb.plot_random_cmc()
    for dataset in "trainset","testset" :

        distance_matrix = distance.get_distance_matrix(file_tag=dataset,medhod_name="ged", **kwargs)
        tb.plot_cmc("ged ", dataset, distance_matrix, **kwargs)
        distance_matrix = distance.get_distance_matrix(file_tag=dataset,medhod_name="jaccard", **kwargs)
        tb.plot_cmc("jaccard", dataset, distance_matrix, **kwargs)
        distance_matrix = distance.get_distance_matrix(file_tag=str(layer)+"_"+dataset,medhod_name="gmd", **kwargs)
        tb.plot_cmc("gmd", dataset, distance_matrix, **kwargs)
        distance_matrix = distance.get_distance_matrix(file_tag=str(layer) + "_"+dataset,medhod_name="hungarian", **kwargs)
        tb.plot_cmc("hungarian", dataset, distance_matrix, **kwargs)


from sklearn.decomposition import DictionaryLearning,SparseCoder, PCA
from scipy.spatial import distance_matrix
from sklearn.manifold import MDS as SKMDS, LocallyLinearEmbedding as LLE
from mds import MDS
from sklearn.utils.extmath import svd_flip
from scipy import linalg
def distance_experiment(**kwargs):
    kwargs.update(base_args())
    kwargs["device"] =torch.device('cpu')

    kwargs.update(use_smiles(kwargs))
    #kwargs.update(use_bbbp(kwargs))
    #kwargs.update(use_AIDS(kwargs))

    kwargs["knn_metrics"] = ["acc", "aur", "kl"]
    kwargs["name"] += "_exp"
    kwargs["reset_storage"] = False
    model = get_model(**kwargs)
    kwargs["trainset"] = Dl(kwargs["trainset"].dataset, batch_size=1, shuffle=False)
    kwargs["testset"] = Dl(kwargs["testset"].dataset, batch_size=1, shuffle=False)

    caption = kwargs["name"] + " train"
    layer =4

    train_data = get_trace(model, "trainset", **kwargs)
    test_data = get_trace(model, "testset", **kwargs)

    train_graph_list = distance.get_graph_list(train_data[layer])
    test_graph_list = distance.get_graph_list(test_data[layer])

    distances_train = distance.get_distances(distance.graph_distances_matrix, train_graph_list, train_graph_list,
                                  file_tag=str(layer) + "_" + "trainset", medhod_name="gmd", **kwargs)

    distances_train_test = distance.get_distances(distance.graph_distances_matrix, train_graph_list, test_graph_list,
                                        file_tag=str(layer) + "_" + "testset", medhod_name="gmd", **kwargs)
    distances_test = distance.get_distances(distance.graph_distances_matrix, test_graph_list, test_graph_list,
                                        file_tag=str(layer) + "_special_" + "testset", medhod_name="_gmd", **kwargs)
    l = 10



    val_p, vec_p = np.linalg.eigh(distances_train)
    #x = np.dot(np.dot(vec_p,  np.identity(n=len(val_p))* val_p), vec_p.transpose())
    X = vec_p * np.sqrt(val_p, dtype=complex)


    pca = PCAlocal(X, len(X))
    X_train_t = pca.transform(X)
    X_train = pca.reverse_transform(X_train_t)

    X2 = np.dot(X, distances_train_test.transpose())

    X_test = pca.reverse_transform(pca.transform(X2.transpose()))

    revU = pca.reverse_transform(pca.U)

    #print(explained_variance_ratio_)


    """model = MDS(n_components=2 ** k,
                   dissimilarity="precomputed")
    v_all = model.fit_transform(alldistances,mask)
    #v11 = model.fit_transform(distances[:l,:l])
    d_train = distance_matrix(v_train,v_train)
    d_all = distance_matrix(v_all,v_all)
    d_train2 = d_all[:a,:a]
    d_test2 = d_all[a:,:a]
    d_test = d_all[a:,a:]"""



    def calc_distance(a,b):
        return np.linalg.norm(a-b)**2/(a.shape[0]*a.shape[1])
    print(calc_distance(np.real(X@X.T),distances_train))
    print(calc_distance(np.real(np.dot(X_train, X_train.transpose())), distances_train))
    print(calc_distance(distance_matrix(X_train, X_train),distances_train))

    print(calc_distance(np.real(np.dot(revU, revU.transpose())), distances_train))
    print(calc_distance(distance_matrix(revU,revU),distances_train))

    print(calc_distance(distance_matrix(X_train_t, X_train_t),distances_train))
    print(calc_distance(np.real(np.dot(X_test, X_test.transpose())), distances_test))
    #print(calc_distance(d_test, distances_test))
    """print(calc_distance(d_train,distances_train))
    print(calc_distance(d_all,alldistances))
    print(calc_distance(d_train2, dt2[:a,:a]))
    print(calc_distance(d_test2,distances_train_test[:,:a]))
    print(calc_distance(d_test, distances_test))"""
    #print(np.linalg.norm(d_all-alldistances)**2/(alldistances.shape[0]*alldistances.shape[1]))


    print("")




def get_orthogonal_vectors(vectors):
    vec= vectors.transpose()/np.linalg.norm(vectors, axis =1)
    disimilarities = np.abs(np.dot(vec.transpose(),vec))
    indexes = orho_base(disimilarities, profondeur=vectors.shape[1])

    return indexes

def orho_base(similarities, profondeur):
    similarities= similarities-np.min(similarities)
    best = similarities.sum()
    for i in range(len(similarities[:10])):
        inds, sum = ortho_rec([i], similarities, profondeur, 0, best)
        if sum < best:
            best = sum
            return_indexes = inds

    return return_indexes

def ortho_rec(indexes, simimilarities, profondeur, sum, upper_bound):
    if len(indexes) >= profondeur:
        return indexes, sum
    if sum > upper_bound:
        return indexes, sum
    sim = simimilarities[indexes].sum(axis=0)
    ids=sim.argsort()
    #ids = list(filter(lambda x: x>indexes[-1], ids))[:4]

    ids = list(filter(lambda x: x not in indexes, ids))[:np.random.randint(1,3)]

    #ids = list(filter(lambda x: sim[x]<sim[ids[0]]*2,ids))
    ret_ind = indexes
    for id in ids:
    #for id in range(indexes[-1]+1, len(sim)):
        ind,  s = ortho_rec(indexes + [id], simimilarities,profondeur, sum+sim[id], upper_bound)
        if s <upper_bound:
            upper_bound = s
            ret_ind = ind
    return ret_ind, upper_bound

class PCAlocal:
    def __init__(self, data, n_dim):
        self.n_components = n_dim
        self.mean = np.mean(data, axis=0)
        X = data - self.mean
        U, S, V = linalg.svd(X, full_matrices=False)
        U, V = svd_flip(U, V)

        n_samples = len(X)
        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        self.explained_variance_ratio_ = explained_variance_ / total_var

        U = U[:, :self.n_components]
        self.S=S
        self.U = U * S[:self.n_components]
        self.components = V#[:self.n_components]
        #reversed = np.dot(U, V[:n_components]) + mean



    def transform(self, X):
        return np.dot(X - self.mean,  self.components.transpose())[:,:self.n_components]

    def reverse_transform(self, X):
        return np.dot(X, self.components[:self.n_components]) + self.mean


import utils.gnn_explainer

from subgroup import getrules,get_dis_rules,frequent_motifs, get_sum_rules,get_decision_tree,gen_rules,workflow,plot_rules
def rules(**kwargs):
    #csv_to_df()

    kwargs.update(base_args())
    kwargs["device"] = torch.device('cpu')

    #kwargs.update(use_smiles(kwargs))
    kwargs.update(use_bbbp(kwargs))
    #kwargs.update(use_AIDS(kwargs))
    #
    gen_bbbp_pkl(kwargs["trainset"])
    #kwargs.update(use_S_clique(kwargs, xpn=0))

    kwargs["name"] += "_rules"
    kwargs["reset_storage"] = True

    model = get_model(**kwargs)
    kwargs["trainset"] = Dl(kwargs["trainset"].dataset, batch_size=1, shuffle=False)
    kwargs["testset"] = Dl(kwargs["testset"].dataset, batch_size=1, shuffle=False)

    train_data = get_trace(model, "trainset", **kwargs)
    test_data = get_trace(model, "testset", **kwargs)

    #utils.gnn_explainer.gnn_explainer_bridge(model, **kwargs)

    layer = len(kwargs["hidden_dims"])
    #extract_csv(train_data, **kwargs)
    #get_adj_mat(**kwargs)
    #get_decision_tree(train_data, **kwargs)
    dl = Dl(kwargs["trainset"].dataset, batch_size=1, shuffle=False)
    model.eval()
    th = get_threshold(model, kwargs["trainset"])
    kwargs["model_output"] = [model(el).detach().numpy()[1]>th for el in dl]

    #extract_csv(train_data, **kwargs)
    #get_adj_mat(**kwargs)
    plot_rules(train_data, **kwargs)

    #workflow(train_data,**kwargs)
    #gen_rules(train_data, **kwargs)

    #
    #getrules(train_data, layer-1, **kwargs)

    #frequent_motifs(train_data, layer, **kwargs)
    #get_sum_rules(train_data, **kwargs)
    #get_decision_tree(train_data, **kwargs)
    #get_dis_rules(train_data, layer, **kwargs)

import pandas as pd

from torch_geometric.utils import to_dense_adj
import pickle as pkl

def get_adj_mat(**kwargs):
    dset= Dl(kwargs["trainset"].dataset, batch_size=1)
    dd= [to_dense_adj(edge_index=el.edge_index,edge_attr=el.edge_attr, max_num_nodes=el.x.size()[0]) for el in dset]

    filename = "bbbp_mat.pkl"
    fileObject = open(filename, 'wb')

    pkl.dump(dd, fileObject)
    fileObject.close()

def load_adj_mat():
    filename = "bbbp_mat.pkl"

    file = open(filename, 'rb')
    matr = pkl.load(file)
    file.close()
    return matr

import subgroup
def extract_csv(data, **kwargs):
    dframe = subgroup.to_dataframe(data, **kwargs)
    dd = subgroup.relabel_model_output(dframe,**kwargs)

    dset = kwargs["trainset"].dataset
    """
    elements=  [dset.dataset[el] for el in dset.indices]
    atoms = [np.argmax(x.numpy()) for el in elements for x in el.x]

    layersize = kwargs["hidden_dims"][0]
    nlayer = len(kwargs["hidden_dims"])
    d2 = np.array([[data[0][el][1]]+ [x  for l in range(nlayer) for x in data[l][el][0][0].tolist()] for el in range(len(data[0])) ])
    DD = pd.DataFrame(d2, columns=["id"]+["l_"+ str(i//layersize) +"c_"+ str(i%layersize) for i in range(nlayer*len(data[0][0][0][0]))])
    DD["atoms"]= atoms
    labels = pd.DataFrame({"class": [kwargs["trainset"].dataset[el[1]].y.numpy()[0] for el in data[0] ]})
    DD = pd.concat([DD, labels], axis=1)"""
    #f = open('datasets/trace/'+kwargs["name"]+"csv", 'w')
    name = 'datasets/trace/'+kwargs["name"]
    dd.to_csv(name+".csv", compression="infer",float_format='%.3f')
    pd.Series(dset.indices).to_csv(name+ "indices.csv", compression="infer")


def csv_to_df():
    """
    le dataframe a les colonnes només "l_xc_y"
    avec x le numéro de la couche et y composate du feature vector
    il y a id qui est l'id de la molécule
    il y a atoms qui est un entier reférent a l'atome
    class, 0 ou 1 selon la classe


    :return:
    """
    path = "datasets/trace/"
    file_name = "syn1_('ba', ('house',))m_5_rules"
    #file_name = "aids_rules"
    #file_name = "hedonicity_rules"
    df = pd.read_csv(path+file_name+".csv")
    s =pd.read_csv(path+file_name+"indices.csv")

    "les valeurs sont des flotants (la valeur du feature vector) pour passer a des booleen (activation) il faut faire ceci:"
    df.update(df.filter(regex="l_[0-9]*c_[0-9]*")>0)


    return df,s
import pickle as pkl

def gen_bbbp_pkl(dataset):
    dset= Dl(dataset.dataset, batch_size=1)

    adj= [to_dense_adj(edge_index=el.edge_index,edge_attr=el.edge_attr, max_num_nodes=el.x.size()[0]).numpy().tolist() for el in dset]
    adj= [list(zip(*el.edge_index.numpy().tolist()))for el in dset]

    feat = [np.argmax(el.x.numpy(),axis=1).tolist() for el in dset]
    labels =[el.y.numpy()[0] for el in dset]

    l=1
    node_map=list()
    with open("datasets/pkls/BBBP_graph_indicator.txt", 'w') as fin:
        for i, nodes in enumerate(feat):
            [fin.write(str(1+i)+"\n") for _ in nodes]
            node_map.append([l+i for i in range(len(nodes))])
            l+=len(nodes)


    with open("datasets/pkls/BBBP_A.txt", 'w') as fin:
        for i, g in enumerate(adj):
            [fin.write(str(node_map[i][a])+", " +str(node_map[i][b])+"\n") for a, b in g]

    with open("datasets/pkls/BBBP_graph_labels.txt", 'w') as fin:
        [fin.write(str(l)+"\n") for l in labels]

    with open("datasets/pkls/BBBP_node_labels.txt", 'w') as fin:
        for el in feat:
            [fin.write(str(l)+"\n") for l in el]

    with open("datasets/pkls/BBBP_edge_gt.txt", 'w') as fin:
        for g in adj:
            [fin.write("0\n") for _ in g]

    print(adj)