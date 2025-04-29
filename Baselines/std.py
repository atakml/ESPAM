import pickle 
import numpy as np

def load_data(dataset):
    if False and dataset == "BBBP":
        with open("BBBP_res_dict.pkl", "rb") as file:
            data = pickle.load(file)
        return data
    total_dict = {}
    for i in range(4):
        try: 
            with open(f"{dataset.lower()}_res_svx_{i}.pkl", "rb") as file:
                data = pickle.load(file)
        except:
            if not i:
                with open(f"{dataset}_svx_dict.pkl", "rb") as file:
                    data = pickle.load(file)
            else:
                raise
        total_dict.update(data)
    return total_dict

for dataset in ["aids", "ba2", "BBBP"]:
    for explainer in ["SVX"]:#['PGExplainer', 'GNNExplainer', 'GradCAM', 'PGM']:
        init_values = {"SVX":{"aids": 0.532, "ba2": 0.577, "BBBP": 0.553}, "subgraph":{"aids":0.496, "ba2":0.517, "bbbp":0.502}}
        res = [init_values[explainer][dataset]]      
        data = load_data(dataset)
        for i in range(4):
            #print(data)
            #print(data[(explainer,i)])
            res.append(data[(explainer,i)].item())


        mean = np.mean(res)
        std = np.std(res)


        print(dataset, explainer)
        print(f"Mean ± Std: {mean:4f} ± {std:4f}")
