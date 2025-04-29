import jax
from jax import numpy as jnp

from GNN_Transformer.utils import tf_to_jax

from GNN_Transformer.make_loss_func import make_loss_func
from GNN_Transformer.make_compute_metrics import make_compute_metrics

def make_eval_step():
    def eval_step(state, batch):
        logits = state.apply_fn(state.params, batch[:-1], deterministic = True)
        return logits
    # return eval_step
    return jax.jit(eval_step)


def make_valid_epoch(loss_option, logger, loader_output_type = 'jax'):
    """
    Helper function to create valid_epoch function.
    """
    compute_metrics = make_compute_metrics(is_weighted = False, loss_option = loss_option, use_jit = True)
    eval_step = make_eval_step()
    # Case loader outputs jnp.DeviceArray:
    if loader_output_type == 'jax':
        def valid_epoch(state, valid_loader):
            batch_metrics = []
            for i, batch in enumerate(valid_loader):
                seq = batch[0]
                # mols, line_mols = batch[1]
                G = batch[1]
                labels = batch[2]
                if isinstance(labels, (list, tuple)):
                    labels = labels[0]
                S = seq # ['hidden_states']
                batch = (S, G, labels)
                # batch = (S, mols, line_mols, labels)
                # batch = flax.jax_utils.replicate(batch)
                logits = eval_step(state, batch)
                metrics = compute_metrics(logits, labels = labels)
                logger.debug('eval_step: {}:  eval_loss:  {}'.format(i, metrics['loss']))
                batch_metrics.append(metrics)
            valid_loader.reset()
            return batch_metrics
    # Case loader outputs tf.Tensor:
    elif loader_output_type == 'tf':
        def valid_epoch(state, valid_loader):
            batch_metrics = []
            for i, batch in valid_loader.enumerate():
                batch = jax.tree_map(lambda x: jax.device_put(tf_to_jax(x), device = jax.devices()[0]), batch)
                seq = batch[0]
                # mols, line_mols = batch[1]
                G = batch[1] 
                labels = batch[2]
                if isinstance(labels, (list, tuple)):
                    labels = labels[0]
                S = seq # ['hidden_states']
                batch = (S, G, labels)
                # batch = (S, mols, line_mols, labels)
                # batch = flax.jax_utils.replicate(batch)
                logits = eval_step(state, batch)
                metrics = compute_metrics(logits, labels = labels)
                logger.debug('eval_step: {}:  eval_loss:  {}'.format(i, metrics['loss']))
                batch_metrics.append(metrics)
            return batch_metrics
    return valid_epoch