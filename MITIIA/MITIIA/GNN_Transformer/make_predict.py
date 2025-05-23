import jax
from jax import numpy as jnp

from GNN_Transformer.utils import tf_to_jax
import code

def make_predict_step(return_intermediates = False):
    if return_intermediates:
        def predict_step(state, batch):
            logits, intermediates = state.apply_fn(state.params, batch, deterministic = True, mutable=['intermediates'])
            pred_probs = jax.nn.sigmoid(logits)
            return pred_probs, intermediates
    else:
        def predict_step(state, batch):
            logits = state.apply_fn(state.params, batch, deterministic = True)
            pred_probs = jax.nn.sigmoid(logits)
            return pred_probs
    # return predict_step
    return jax.jit(predict_step)


def make_predict_epoch(logger, loader_output_type = 'jax', return_intermediates = False):
    """
    Helper function to create predict_epoch function.
    """
    predict_step = make_predict_step(return_intermediates = return_intermediates)
    # Case loader outputs jnp.DeviceArray:
    if loader_output_type == 'jax':
        def predict_epoch(state, predict_loader):
            batch_predictions = []

            for i, batch in enumerate(predict_loader):
                seq = batch[0]
                G = batch[1] 
                labels = batch[2]

                if isinstance(labels, (list, tuple)):
                    labels = labels[0]
                S = seq # ['hidden_states']
                batch = (S, G)
                # batch = flax.jax_utils.replicate(batch)
                pred_probs = predict_step(state, batch)
                batch_predictions.append(pred_probs)
            if not isinstance(predict_loader, list):
                predict_loader.reset()
            return batch_predictions
    # Case loader outputs tf.Tensor:
    elif loader_output_type == 'tf':
        def predict_epoch(state, predict_loader):
            batch_predictions = []
            for i, batch in enumerate(predict_loader):
                batch = jax.tree_map(lambda x: jax.device_put(tf_to_jax(x), device = jax.devices()[0]), batch)
                seq = batch[0]
                G = batch[1]
                labels = batch[2]
                if isinstance(labels, (list, tuple)):
                    labels = labels[0]
                S = seq # ['hidden_states']
                batch = (S, G)
                # batch = flax.jax_utils.replicate(batch)
                pred_probs = predict_step(state, batch)
                batch_predictions.append(pred_probs)
            return batch_predictions
    return predict_epoch


# ---------------
# predict single:
# ---------------
def make_apply_bert(bert_model):
    """
    Take the last 5 CLS layers.
    """
    # @jax.jit
    def apply_bert(seq):
        bert_output = bert_model.module.apply({'params': bert_model.params}, **seq, deterministic = True,
                             output_attentions = False,
                             output_hidden_states = True, 
                             return_dict = True)
        S = bert_output.hidden_states
        S = jnp.stack(S[-5:], axis = 1)
        S = jnp.reshape(S[:, :, 0, :], newshape = (S.shape[0], -1))
        return S
    return apply_bert

def make_predict_single_epoch(logger, bert_model, loader_output_type = 'jax', return_intermediates = False):
    """
    Helper function to create predict_epoch function.
    """
    predict_step = make_predict_step(return_intermediates = return_intermediates)
    apply_bert = make_apply_bert(bert_model)
    # Case loader outputs jnp.DeviceArray:
    if loader_output_type == 'jax':
        def predict_epoch(state, predict_loader):
            batch_predictions = []
            print(predict_loader)
            for i, batch in enumerate(predict_loader):
                seq = batch[0]
                G = batch[1]
                print(jax.tree_map(lambda x: x.shape, G))
                print(G.globals['node_padding_mask'])
                # raise Exception('Fuuck...') 
                labels = batch[2]
                if isinstance(labels, (list, tuple)):
                    labels = labels[0]
                S = apply_bert(seq) # Apply BERT
                batch = (S, G)
                # batch = flax.jax_utils.replicate(batch)
                output = predict_step(state, batch)
                batch_predictions.append(output)
            predict_loader.reset()
            return batch_predictions
    # Case loader outputs tf.Tensor:
    elif loader_output_type == 'tf':
        def predict_epoch(state, predict_loader):
            batch_predictions = []
            for i, batch in predict_loader.enumerate():
                batch = jax.tree_map(lambda x: jax.device_put(tf_to_jax(x), device = jax.devices()[0]), batch)
                seq = batch[0]
                G = batch[1]
                print(jax.tree_map(lambda x: x.shape, G))
                print(G.globals['node_padding_mask'])
                # raise Exception('Fuuck...') 
                labels = batch[2]
                if isinstance(labels, (list, tuple)):
                    labels = labels[0]
                S = apply_bert(seq) # Apply BERT
                batch = (S, G)
                # batch = flax.jax_utils.replicate(batch)
                output = predict_step(state, batch)
                batch_predictions.append(output)
            return batch_predictions
    return predict_epoch
