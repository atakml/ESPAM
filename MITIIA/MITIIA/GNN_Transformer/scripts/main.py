import os
import yaml
import json
import datetime
import argparse

from envyaml import EnvYAML


def prepare_params(config_file):
    env = EnvYAML(config_file, flatten=False)
    params = env#.yaml_config

    # Read params not using envyaml library (without parsing environment variables):
    # with open(args.config, 'r') as yamlfile:
    #     params = yaml.safe_load(yamlfile)

    # Cast ATOM_FEATURES and BOND_FEATURES to tuple.
    #print(params['ATOM_FEATURES'])
    #print(tuple(params['ATOM_FEATURES']))
    #params['ATOM_FEATURES'] = tuple(params['ATOM_FEATURES'])
    #params['BOND_FEATURES'] = tuple(params['BOND_FEATURES'])
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='config file path')
    parser.add_argument('--cuda_device', type=int,
                        help='Set environment variable CUDA_VISIBLE_DEVICES')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to a directory where output is saved')

    args = parser.parse_args()
    print('Config file: {}'.format(args.config))
    print('---------------')

    # Set visible devices:
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        print('Setting CUDA_VISIBLE_DEVICES to: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    params = prepare_params(args.config)

    # Consistency checks:
    if params['SELF_LOOPS']:
        if not params['MODEL_NAME'] in ['Simple_GAT_model', 'Transformer_GAT_model']:
            raise ValueError('Not supported model for self loops: {}'.format(params['MODEL_NAME']))
        if len(params['BOND_FEATURES']) > 0:
            raise ValueError('Non-empty bond features while self loop is True.')

    if 'SIZE_CUT_DIRNAME' in params.keys():
        if params['ACTION'] != 'train' and params['SIZE_CUT_DIRNAME'] is not None:
            raise NotImplementedError('Size cut is allowed only for training for now.')

    # Choose function to run:
    if params['CASE'] == 'BERT_GNN':
        if params['ACTION'] == 'train':
            from GNN_Transformer.main_train import main_train

            output = main_train(params)
        elif params['ACTION'] == 'eval':
            from GNN_Transformer.main_eval import main_eval

            output = main_eval(params)
        elif params['ACTION'] == 'predict':
            from GNN_Transformer.main_predict import main_predict

            output = main_predict(params)
        elif params['ACTION'] == 'predict_single':
            from GNN_Transformer.main_predict_single import main_predict_single

            output = main_predict_single(params)
        else:
            raise ValueError(
                'Unknown action {}. Available options: {}'.format(params['ACTION'], ['train', 'eval', 'predict']))
    print('Finished...')

    if args.output_dir is not None and output is not None:
        _datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(os.path.join(args.output_dir,
                               params['ACTION'] + '_output_' + params['MODEL_NAME'] + '_' + _datetime + '.json'),
                  'w+') as jsonfile:
            json.dump(output, jsonfile)

    # for key in params.keys():
    #     print(key, ':', params[key])
