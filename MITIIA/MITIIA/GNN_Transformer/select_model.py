# CLS_GTransformer:
from GNN_Transformer.model.dummy import DummyModel
from GNN_Transformer.model.normal_GraphOnly_mhaOnly_QK_large import CLS_GTransformer_normal_GraphOnly_mhaOnly_QK_large_model


def get_model_by_name(name):
    """
    Utils to retrieve model given its name
    """
    # CLS_GTransformer:
    if name == 'DummyModel':
        model_class = DummyModel
    elif name == 'CLS_GTransformer_normal_GraphOnly_mhaOnly_QK_large_model':
        model_class = CLS_GTransformer_normal_GraphOnly_mhaOnly_QK_large_model
    
    # Other:
    else:
        raise ValueError('Unknown model name: {}'.format(name))

    return model_class    


