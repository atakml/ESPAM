import torch

import dataloader
from metrics import mymse, binary_acc, bin_prec, bin_rec, au_roc, au_pr,optimal_acc,f1_score_metric
import torch_geometric
import numpy as np
import random
import model

def is_cuda_available(gpu=True):
    if gpu and torch.cuda.is_available():
        print("cuda enabled")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def base_args():
    seed = 3
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    kwargs = {
        'device': is_cuda_available(),
        'verbose': False,
        'epochs': 70,

        'batch_size': 512,
        'dataset_split_ratio': 0.8,

        'loss_weight': torch.tensor([0.337, 0.663]),
        # 'loss_weight': torch.tensor([0.75,0.25]),
        # 'loss_weight': torch.tensor([0.2, 0.8]),

        'lr': 0.01,
        'weight_decay': 0.0001,
        'betas': [0.9, 0.999],

        'model_dir': "models/",
        'model_save': "model_file",

        'new_data_dir': "feature_vectors",
        'new_data': 'new_data',
        'reset_storage': False,
        "get_metrics": False,
        "seed" : seed
    }
    args = {
        'nc': 256,

        # 'metrics': (torch.nn.MSELoss(), torch.nn.BCEWithLogitsLoss(), min_rank, max_rank,  min_accuracy,precision(i), recall(i), n_th_accuracy(i)),
        'metrics': (
            mymse(),
            torch.nn.CrossEntropyLoss(weight=kwargs["loss_weight"]),
            binary_acc,
            bin_prec,
            bin_rec,
            optimal_acc,
            f1_score_metric,
            # bin_rev_prec,
            # bin_rev_rec,
            au_roc,
            au_pr),
        # 'loss': torch.nn.CrossEntropyLoss(weight=kwargs["loss_weight"].to(kwargs["device"])),
        'loss': mymse(),
        'print_all': True

    }
    kwargs.update(args)
    train, test = dataloader.tox_21_dl("datasets", "Tox21_AhR_training", "Tox21_AhR_testing", balance=None, **kwargs)
    args = {
        "trainset": train,
        "testset": test,
        "input_dim": train.dataset.num_features,
        "output_dim": train.dataset.num_classes,
        'hidden_dims': [30] * 6,
        'decoder_dims': []

    }
    kwargs.update(args)
    return kwargs


def use_smiles(kwargs):
    kwargs["label_filter"] = [23]
    train, test = dataloader.molnet_dataloader("hedonicity", dataloader.MultiLabelTransform([23]), 0.8, clean=False, **kwargs)
    args = {
        "name" : "hedonicity",
        "epochs": 80,
        "trainset": train,
        "testset": test,
        "input_dim": train.dataset[0].num_features,
        "output_dim": 2,
        'hidden_dims': [35] * 5,
        'decoder_dims': [],
        'new_data': 'new_data_hedo',

    }
    kwargs["atoms"] = dataloader.atoms
    kwargs["indexes"] = dataloader.indexes
    kwargs.update(args)
    return kwargs


def use_smiles_edonicity(kwargs):
    kwargs["label_filter"] = [23]
    train, test = dataloader.molnet_dataloader("hedonicity", dataloader.HedonicityValueTransform(), 0.8, **kwargs)
    args = {
        "name" : "hedonicity_reg",

        "epochs": 80,
        "trainset": train,
        "testset": test,
        "input_dim": train.dataset[0].num_features,
        "output_dim": 1,
        'hidden_dims': [30] * 5,
        'decoder_dims': [],
        'model_save': "model_file_edo",
        'lr': 0.01,
        'weight_decay': 0.0001,
        'new_data_dir': "feature_vectors",
        'new_data': 'new_data_edo',
        'metrics': (torch.nn.MSELoss(), torch.nn.L1Loss()),
        'loss': torch.nn.MSELoss()
    }
    kwargs.update(args)
    return kwargs


def use_bbbp(kwargs):
    train, test = dataloader.molnet_dataloader("BBBP", None, 0.8,  **kwargs)
    kwargs["loss_weight"]=torch.tensor([0.7,0.3])
    args = {
        "name" : "BBBP",
        "trainset": train,
        "testset": test,
        "input_dim": train.dataset[0].num_features,
        "output_dim": 2,
        'hidden_dims': [20] * 6,
        'decoder_dims': [],
        'loss': torch.nn.CrossEntropyLoss(weight=kwargs["loss_weight"].to(kwargs["device"])),
        'new_data': 'new_data_bbbp',

        #'metrics': (torch.nn.CrossEntropyLoss(weight=kwargs["loss_weight"]), torch.nn.MSELoss()),
    }

    kwargs["atoms"] = dataloader.atoms
    kwargs["indexes"] = dataloader.indexes
    kwargs.update(args)
    return kwargs


def use_AIDS(kwargs):
    train, test = dataloader.aids_dl(dataloader.BASE_PATH, 0.8, **kwargs)
    kwargs["loss_weight"]=torch.tensor([0.7,0.3])
    args = {
        "name": "aids",
        "trainset": train,
        "testset": test,
        "input_dim": train.dataset[0].num_features,
        "output_dim": 2,
        'hidden_dims': [15] * 5,
        'decoder_dims': [],
        'loss': torch.nn.CrossEntropyLoss(weight=kwargs["loss_weight"].to(kwargs["device"])),
        'new_data': 'new_data_aids',

        #'metrics': (torch.nn.CrossEntropyLoss(weight=kwargs["loss_weight"]), torch.nn.MSELoss()),
    }
    kwargs["atoms"]= ["C" , "O" , "N" , "Cl", "F" , "S" , "Se", "P" , "Na", "I" , "Co", "Br", "Li", "Si", "Mg", "Cu", "As", "B" , "Pt", "Ru", "K" , "Pd", "Au", "Te", "W" , "Rh", "Zn", "Bi", "Pb", "Ge", "Sb", "Sn", "Ga", "Hg", "Ho", "Tl", "Ni", "Tb"]
    kwargs["indexes"]= {a: i for i, a in enumerate(kwargs["atoms"])}

    kwargs.update(args)
    return kwargs

def use_SDNE(kwargs):
    args = {
        "encoder_dims":[20,15],
        "embedding_dim": 10,
        "decoder_dims": [15,20],
        "metrics" :  (model.SDNE.mseloss, model.SDNE.loss_1st)
    }
    kwargs.update(args)
    return kwargs

def use_S_clique(kwargs, xpn=0):
    shapes = [
        {"shape":("ba", ("house",)), "rdm_params":(6,15), "m": 5, "rdm_basis":True},
        {"shape":("tree", ("cycle",5)), "rdm_params":(2,8), "m": 5, "rdm_basis":True},
        {"shape":("ba", ("house",)), "rdm_params":(6,20), "m": 1, "rdm_basis":True}
    ]


    kwargs["batch_size"]=20
    train, test = dataloader.synth_dataloader(shapes[xpn],  400, 0.8, **kwargs)
    kwargs["loss_weight"]=torch.tensor([0.5,0.5])
    args = {
        "name": "syn1_"+str(shapes[xpn]["shape"])+"m_"+str(shapes[xpn]["m"]),
        "epochs": 50,
        "trainset": train,
        "testset": test,
        "input_dim": train.dataset[0].num_features,
        "output_dim": 2,
        'hidden_dims': [10] * 5,

        'decoder_dims': [],
        'loss': torch.nn.CrossEntropyLoss(),
        'new_data': 'new_data_clique',
        'lr': 0.005,
        'weight_decay': 0.001,
        'betas': [0.9, 0.999],

        #'metrics': (torch.nn.CrossEntropyLoss(weight=kwargs["loss_weight"]), torch.nn.MSELoss()),
    }
    kwargs["atoms"]= ["X" for _ in range(train.dataset[0].num_features) ]#["C" , "O" , "N" , "Cl", "F" , "S" , "Se", "P" , "Na", "I" , "Co", "Br", "Li", "Si", "Mg", "Cu", "As", "B" , "Pt", "Ru", "K" , "Pd", "Au", "Te", "W" , "Rh", "Zn", "Bi", "Pb", "Ge", "Sb", "Sn", "Ga", "Hg", "Ho", "Tl", "Ni", "Tb"]
    kwargs["indexes"]= {a: i for i, a in enumerate(kwargs["atoms"])}

    kwargs.update(args)
    return kwargs


import argparse




def arg_parse():
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    parse_optimizer(parser)

    parser.add_argument("--clean-log", action="store_true", help="If true, cleans the specified log directory before running.")
    parser.add_argument("--logdir", dest="logdir", help="Tensorboard log directory")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=False,
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--no-writer",
        dest="writer",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    # Explainer
    parser.add_argument("--mask-act", dest="mask_act", type=str, help="sigmoid, ReLU.")
    parser.add_argument(
        "--mask-bias",
        dest="mask_bias",
        action="store_const",
        const=True,
        default=False,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--explain-node", dest="explain_node", type=int, help="Node to explain."
    )
    parser.add_argument(
        "--graph-idx", dest="graph_idx", type=int, help="Graph to explain."
    )
    parser.add_argument(
        "--graph-mode",
        dest="graph_mode",
        action="store_const",
        const=True,
        default=False,
        help="whether to run Explainer on Graph Classification task.",
    )
    parser.add_argument(
        "--multigraph-class",
        dest="multigraph_class",
        type=int,
        help="whether to run Explainer on multiple Graphs from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--multinode-class",
        dest="multinode_class",
        type=int,
        help="whether to run Explainer on multiple nodes from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--align-steps",
        dest="align_steps",
        type=int,
        help="Number of iterations to find P, the alignment matrix.",
    )

    parser.add_argument(
        "--method", dest="method", type=str, help="Method. Possible values: base, att."
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )
    parser.add_argument(
        "--explainer-suffix",
        dest="explainer_suffix",
        help="suffix added to the explainer log",
    )

    # TODO: Check argument usage
    parser.set_defaults(
        logdir="log",
        ckptdir="ckpt",
        dataset="syn1",
        opt="adam",
        opt_scheduler="none",
        cuda="0",
        lr=0.1,
        clip=2.0,
        batch_size=20,
        num_epochs=100,
        hidden_dim=20,
        output_dim=20,
        num_gc_layers=3,
        dropout=0.0,
        method="base",
        name_suffix="",
        explainer_suffix="",
        align_steps=1000,
        explain_node=None,
        graph_idx=-1,
        mask_act="sigmoid",
        multigraph_class=-1,
        multinode_class=-1,
    )
    return parser.parse_args()


def parse_optimizer(parser):
    '''Set optimizer parameters'''
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
            help='Type of optimizer scheduler. By default none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
            help='Number of epochs before decay')
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
            help='Learning rate decay ratio')
    opt_parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')