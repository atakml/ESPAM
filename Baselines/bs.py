#Selector, load_dataset, get_classification_task, model_selector
from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.tasks.replication import get_classification_task, to_torch_graph
from ExplanationEvaluation.models.model_selector import model_selector
import torch

from torch_geometric.data import Data

_dataset = "bbbp"

_explainer = "actexplainer"

_folder = 'replication' # One of: replication, extension

config_path = f"ExplanationEvaluation/configs/{_folder}/explainers/{_explainer}/{_dataset}.json"
print(config_path)
radius = 3
config = Selector(config_path)
print(config)
config = config.args.explainer

graphs, features, labels, train_mask, _, test_mask = load_dataset(config.dataset)
task = get_classification_task(graphs)

features = torch.tensor(features)
labels = torch.tensor(labels)
graphs = to_torch_graph(graphs, task)



# Load pretrained models
model, checkpoint = model_selector(config.model,
                                    config.dataset,
                                    pretrained=True,
                                    return_checkpoint=True)
if config.eval_enabled:
    model.eval()

import numpy as np
import os
import time
cnt = 0
import pickle
import subprocess
import warnings
warnings.filterwarnings('ignore')
def count_processes_with_command(command):
    # Get the list of processes
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    # Filter processes that match the command
    process_list = result.stdout.splitlines()
    count = sum(command in line for line in process_list)
    return count

explainer = "subgraph"


# Specify the command you're looking for
command = "python prc.py" if explainer == "subgraph" else "python proc.py"

#device = torch.cuda.current_device()

from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
train_indices = np.where(train_mask)[0]
while count_processes_with_command("python bs.py") > 1:
    print("sleeping for 100s")
    time.sleep(100)

for index in train_indices:
    try:
        with open(f"{_dataset}/{explainer}/{index}.pkl", "rb") as file:
            #pickle.dump(dataset.graphs[index], file)
            pickle.load(file)
    except:
        with open(f"{_dataset}/{explainer}index/{index}.pkl", "wb") as file:
            data = Data(edge_index=graphs[index], x = features[index])
            pickle.dump(data, file)
    else: 
        continue
    os.system(f"{command} {_dataset} --i {index} &")
    number_of_processes = count_processes_with_command(command)
    #h = nvmlDeviceGetHandleByIndex(0)
    #info = nvmlDeviceGetMemoryInfo(h)
    print(count_processes_with_command(command))
    while count_processes_with_command(command) > 1: #or info.free / (1024 ** 3) < 10: #or count_processes_with_command(command) > 1:
        print(f'free     : {info.free/(1024**3)}')
        time.sleep(1)
        #h = nvmlDeviceGetHandleByIndex(0)
        #info = nvmlDeviceGetMemoryInfo(h)

