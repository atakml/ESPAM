from torch_geometric.data import Data
import torch
from rdkit import Chem
from rdkit.Chem import Draw



import jraph
import jax.numpy as jnp

from mol2graph.jraph.convert import smiles_to_jraph
from GNN_Transformer.utils import pad_graph

from configs import hparams

def compute_model_score(data, seq, model, class_index=None):
    if not isinstance(data, str):
        if data.edge_index.shape[1] == 0:
            return 0
        try:
            G = jraph.GraphsTuple(nodes=jnp.array(data.x.detach(), dtype=jnp.float32),
                                  edges=jnp.array(data.edge_attr, dtype=jnp.float32),
                                  receivers=jnp.array(data.edge_index[0], dtype=jnp.int32),
                                  senders=jnp.array(data.edge_index[1], dtype=jnp.int32),
                                  globals=None,
                                  n_node=jnp.array([data.x.shape[0]]),
                                  n_edge=jnp.array([data.edge_index.shape[1]]))
        except:
            print(data.x, data.edge_index)
            raise
        t = model([(seq, pad_graph(G, 32, 64), 1)])
    else:
        #print(len(pad_graph(smiles_to_jraph(data, atom_features=["AtomicNum", "ChiralTag", "Hybridization", "FormalCharge", "NumImplicitHs", "ExplicitValence", "Mass", "IsAromatic"], bond_features=["BondType", "Stereo", "IsAromatic"]), 32, 64).n_node))
        t = model([(seq, pad_graph(smiles_to_jraph(data, atom_features=["AtomicNum", "ChiralTag", "Hybridization", "FormalCharge", "NumImplicitHs", "ExplicitValence", "Mass", "IsAromatic"], bond_features=["BondType", "Stereo", "IsAromatic"]),32, 64),1),])
    t = jnp.array(list(map(lambda x: (x.tolist()[0][0], 1-x.tolist()[0][0]), t)))
    if class_index is None:
        return t
    return t[0].tolist()[class_index]


def from_smiles_to_data(smile):
    if isinstance(smile, str):
        smile = [smile]
    main_data = None
    for sm in smile:
        try:
            jraph_convert = smiles_to_jraph(sm, atom_features=hparams['ATOM_FEATURES'],
                                            bond_features=hparams["BOND_FEATURES"], self_loops=False)
        except:
            print(sm)
            debug_mol = Chem.MolFromSmiles(sm)
            Draw.MolsToGridImage([debug_mol], molsPerRow=5, subImgSize=(200, 200))
            raise
        data = Data(x=torch.tensor(jraph_convert.nodes.tolist()),
                    edge_index=torch.tensor((jraph_convert.senders.tolist(), jraph_convert.receivers.tolist())),
                    shape=jraph_convert.n_node.tolist()[0], edge_attr=torch.tensor(jraph_convert.edges.tolist()))
        if main_data is None:
            main_data = data
        else:
            data.edge_index = data.edge_index + main_data.x.shape[0]
            main_data.x = torch.cat((main_data.x, data.x), 0)
            main_data.edge_index = torch.cat((main_data.edge_index, data.edge_index), -1)
            main_data.shape += data.shape
            main_data.edge_attr = torch.cat((main_data.edge_attr, data.edge_attr), 0)
    return main_data
