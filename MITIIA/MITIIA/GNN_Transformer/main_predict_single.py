import os
# os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import functools
import pickle
import json
import time
import pandas
import numpy as np
import jax
from jax import numpy as jnp
import flax
from flax import serialization

from GNN_Transformer.loader import ProtBERTDatasetPrecomputeBERT, ProtBERTLoader, ProtBERTCollatePrecomputeBERT_CLS, \
    ProtBERTDataset, ProtBERTCollate
from GNN_Transformer.make_init import make_init_model, get_tf_specs
from GNN_Transformer.make_create_optimizer import make_create_optimizer
from GNN_Transformer.make_predict import make_predict_single_epoch, make_apply_bert, make_predict_epoch
from GNN_Transformer.select_model import get_model_by_name
from GNN_Transformer.utils import _serialize_hparam

import logging


def main_predict_single(hparams):
    """
    """
    # df = {'mutated_Sequence': [hparams['MUTATED_SEQUENCE']],
    #      '_MolID': [hparams['MOL_ID']],
    #      '_SMILES': [hparams['SMILES']]}
    # df = pandas.DataFrame(df)
    # df['Responsive'] = -1

    model_class = get_model_by_name(hparams['MODEL_NAME'])
    model = model_class(atom_features=hparams['ATOM_FEATURES'], bond_features=hparams['BOND_FEATURES'])

    if hparams['SELF_LOOPS']:
        hparams['PADDING_N_EDGE'] = hparams['PADDING_N_EDGE'] + hparams['PADDING_N_NODE']  # NOTE: Because of self_loops
        if len(hparams['BOND_FEATURES']) > 0:
            raise ValueError('Can not have both bond features and self_loops.')

    logger = logging.getLogger('main_predict_single')
    logger.setLevel(logging.INFO)  # logging.DEBUG
    # logger_stdout_handler = logging.StreamHandler(sys.stdout)
    # logger.addHandler(logger_stdout_handler)

    from transformers import BertTokenizer, BertConfig, FlaxBertModel
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    config = BertConfig.from_pretrained("Rostlab/prot_bert", output_hidden_states=True, output_attentions=False)
    bert_model = FlaxBertModel.from_pretrained("Rostlab/prot_bert", from_pt=True, config=config)

    collate = ProtBERTCollate(tokenizer=BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False),
                              padding_n_node=hparams['PADDING_N_NODE'],
                              padding_n_edge=hparams['PADDING_N_EDGE'],
                              n_partitions=hparams['N_PARTITIONS'],
                              seq_max_length=512,
                              line_graph=hparams['LINE_GRAPH'],
                              )

    # predict_dataset = ProtBERTDataset(data_csv=df,
    #                                  mol_col='_SMILES',
    #                                  seq_col='mutated_Sequence',  # Gene is only sequence id.
    #                                  label_col='Responsive',
    #                                  weight_col=None,
    #                                  atom_features=model.atom_features,
    #                                  bond_features=model.bond_features,
    #                                  line_graph_max_size=hparams[
    #                                                          'LINE_GRAPH_MAX_SIZE_MULTIPLIER'] * collate.padding_n_node,
    #                                  self_loops=hparams['SELF_LOOPS'],
    #                                  line_graph=hparams['LINE_GRAPH'],
    #                                  )

    # _predict_loader = ProtBERTLoader(predict_dataset,
    #                                 batch_size=hparams['BATCH_SIZE'],
    #                                 collate_fn=collate.make_collate(),
    #                                 shuffle=False,  # NOTE: shuffle is redundant for tf.data.Dataset here.
    #                                 rng=None,
    #                                 drop_last=False,
    #                                 n_partitions=hparams['N_PARTITIONS'])

    # if hparams['LOADER_OUTPUT_TYPE'] == 'jax':
    #    predict_loader = _predict_loader
    # elif hparams['LOADER_OUTPUT_TYPE'] == 'tf':
    #    predict_loader = _predict_loader.tf_Dataset(output_signature=get_tf_specs(is_weighted=False,
    #                                                                              n_partitions=hparams['N_PARTITIONS'],
    #                                                                              num_node_features=len(
    #                                                                                  hparams['ATOM_FEATURES']),
    #                                                                              num_edge_features=len(
    #                                                                                  hparams['BOND_FEATURES']),
    #                                                                              padding_n_node=hparams[
    #                                                                                  'PADDING_N_NODE'],
    #                                                                              padding_n_edge=hparams[
    #                                                                                  'PADDING_N_EDGE'],
    #                                                                              line_graph=hparams['LINE_GRAPH'],
    #                                                                              self_loops=hparams['SELF_LOOPS']))
    #    predict_loader = predict_loader.cache()
    #    predict_loader = predict_loader.prefetch(buffer_size=4)
    #    logger.info('loader_output_type = {}'.format(hparams['LOADER_OUTPUT_TYPE']))

    key1, key2 = jax.random.split(jax.random.PRNGKey(int(time.time())), 2)
    key_params, _key_num_steps, key_num_steps, key_dropout = jax.random.split(key1, 4)

    # Initializations:
    start = time.time()
    logger.info('jax_version = {}'.format(jax.__version__))
    logger.info('flax_version = {}'.format(flax.__version__))
    logger.info('Initializing...')
    init_model = make_init_model(model,
                                 batch_size=hparams['BATCH_SIZE'],
                                 seq_embedding_size=1024,
                                 num_node_features=len(hparams['ATOM_FEATURES']),
                                 num_edge_features=len(hparams['BOND_FEATURES']),
                                 self_loops=hparams['SELF_LOOPS'],
                                 line_graph=hparams['LINE_GRAPH'])  # 768)
    params = init_model(rngs={'params': key_params, 'dropout': key_dropout,
                              'num_steps': _key_num_steps})  # jax.random.split(key1, jax.device_count()))
    end = time.time()
    logger.info('TIME: init_model: {}'.format(end - start))

    # This is needed to create state:
    # t = len(predict_dataset)
    t = 37219  # NOTE: hardcoded
    create_optimizer = make_create_optimizer(model, option=hparams['OPTIMIZATION']['OPTION'],
                                             warmup_steps=hparams['OPTIMIZATION']['WARMUP_STEPS'],
                                             transition_steps=hparams['OPTIMIZATION']['TRANSITION_EPOCHS'] * (
                                                     t / hparams['BATCH_SIZE']))
    init_state, scheduler = create_optimizer(params, rngs={'dropout': key_dropout, 'num_steps': key_num_steps},
                                             learning_rate=hparams['LEARNING_RATE'])

    # Restore params:
    if hparams['RESTORE_FILE'] is not None:
        logger.info('Restoring parameters from {}'.format(hparams['RESTORE_FILE']))
        with open(hparams['RESTORE_FILE'], 'rb') as pklfile:
            bytes_output = pickle.load(pklfile)
        state = serialization.from_bytes(init_state, bytes_output)
        logger.info('Parameters restored...')
    else:
        state = init_state

    if hparams['N_PARTITIONS'] > 0:
        raise NotImplementedError('pmap is not implemented because of apply_bert.')
        # state = flax.jax_utils.replicate(state)
        # predict_epoch = make_predict_single_epoch_pmap(logger = logger, loader_output_type = hparams['LOADER_OUTPUT_TYPE'])
    else:
        predict_epoch = make_predict_epoch(logger=logger, loader_output_type=hparams['LOADER_OUTPUT_TYPE'],
                                           return_intermediates=hparams['RETURN_INTERMEDIATES'])

    # Log hyperparams:
    # _hparams = {}
    # for key in hparams.keys():
    #     _hparams[key] = _serialize_hparam(hparams[key])
    # hparams_logs = _hparams
    # with open(os.path.join(datadir, 'Prediction', 'hparams_logs.json'), 'w') as jsonfile:
    #     json.dump(hparams_logs, jsonfile)

    # _dataparams = {}
    # for key in dataparams.keys():
    #     _dataparams[key] = _serialize_hparam(dataparams[key])
    # with open(os.path.join(datadir, 'Prediction', 'dataparams_logs.json'), 'w') as jsonfile:
    #     json.dump(_dataparams, jsonfile)

    # --------
    # PREDICT:
    # --------
    return lambda predict: predict_epoch(state, predict)
    if False:
        start = time.time()
        output = predict_epoch(state, predict_loader)
        end = time.time()
        if hparams['RETURN_INTERMEDIATES']:
            # Attention weights:
            # print(jax.tree_map(lambda x: x.shape, output[0][1]))
            print(output[0][1])
            # Predictions:
            print('Probability: {}'.format(output[0][0]))

            with open(
                    os.path.join(hparams['LOGGING_PARENT_DIR'], hparams['SEQ_ID'] + '__' + hparams['MOL_ID'] + '.pkl'),
                    'wb') as pklfile:
                pickle.dump(flax.core.frozen_dict.unfreeze(output[0][1]), pklfile)
            return output[0][1]
        else:
            print(output)

        if False:
            with open(
                    os.path.join(hparams['LOGGING_PARENT_DIR'], hparams['SEQ_ID'] + '__' + hparams['MOL_ID'] + '.pkl'),
                    'wb') as pklfile:
                pickle.dump(flax.core.frozen_dict.unfreeze(output[0][1]), pklfile)
