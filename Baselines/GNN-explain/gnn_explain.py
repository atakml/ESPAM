import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=400, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=50, help='size of the batches')
parser.add_argument('--n_features', type=int, default=256, help='dimensionality of the featurespace')
parser.add_argument('--nc', type=int, default=256, help='dimensionality of the featurespace')

parser.add_argument('--set_device', type=str, default="cpu", help='set cuda')

parser.add_argument('--save_step', type=int, default=0, help='dir of the trained_model')

parser.add_argument('--save', type=str, default="save", help='trained model output name')
parser.add_argument('--save_dir', type=str, default="experiments_output", help='trained model dir')

parser.add_argument('--lr', type=float, default=0.001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.7, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.8, help='adam: decay of first order momentum of gradient')

parser.add_argument('--verbose', type=bool, default=False, help='adam: decay of first order momentum of gradient')

from experiments import base_test, dataset_balance, base_rule_system,gmd_exp,graph_selection_experiment,retrain_experiment,gnn_ged
from experiments import jaccard_experiment,ged_experiment,distance_experiment, autoencoder_eperiment,gmd_AE,ged_AE,plot_dataset
from experiments import plot_cmc,wl_xperiment
from experiments import rules

#dataset_balance()
#kmean_search()
#test_explain()
#extract_dataset()
#base_rule_system()
#retrain_experiment()
#dataset_balance()
#gmd_exp()
#gmd_AE()
#ged_AE()
rules()
#autoencoder_eperiment()
#plot_dataset()
#plot_cmc()
#distance_experiment()
#wl_xperiment()
#ged_experiment()
#gnn_ged()
#jaccard_experiment()
#retrain_experiment()
#graph_selection_experiment()
#base_test()
#test_new_dataset()

#dataset_balance()
#base_test()
#check()

"""
args = vars(parser.parse_args())
args.update({"metrics": (torch.nn.MSELoss(), torch.nn.BCELoss())})

if args["set_device"] == "cuda" and torch.cuda.is_available():
    args["device"] = torch.device('cuda')
    print("cuda enabled")
else:
    args["device"] = torch.device('cpu')

# train, test = build_smiles_dataset("odorants.csv",6, 0.8, args)
train, test = build_smiles_odors_dataset("qualities.csv", 0.75, **args)

model = GCNet(len(dataloader.atoms), [20] * 6, len(train.dataset[0][2]),loss=torch.nn.BCELoss())

optim = torch.optim.Adam(
    model.parameters(),
    lr=args["lr"],
    weight_decay=0.0005
)
args["print_all"]=True

print(args)
fit(model, optim, train, test, **args)
"""