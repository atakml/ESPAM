import pickle 
import os
from tqdm import tqdm


#for dataset_name in ["ba2motifs", "bbbp"]:
#    for explainer in ["subgraph", "gstar"]:
dataset_name = "bbbp"
explainer = "subgraph"
directory = f"{dataset_name}/{explainer}/"
save_directory = f"/shap_extend/{dataset_name}/"
save_name = f"{explainer}.pkl"
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


merged_dict = {}

for file_name in tqdm(files):
    with open(directory + file_name, "rb") as file:
        data = pickle.load(file)
    merged_dict[int(file_name.split(".")[0])] = data


with open(save_directory + save_name, "wb") as file:
    pickle.dump(merged_dict, file)

