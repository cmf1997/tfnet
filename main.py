#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : main.py
@Time : 2023/11/09 11:18:29
@Author : Cmf
@Version : 1.0
@Desc : None
'''

# here put the import lib
import click
import numpy as np
from functools import partial
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from logzero import logger

from tfnet.data_utils import *
from tfnet.datasets import TFBindDataset
from tfnet.models import Model
from tfnet.evaluation import output_eval, output_predict, CUTOFF

import pdb


# code
def train(model, data_cnf, model_cnf, train_data, valid_data=None, class_weights_dict = None, random_state=1240):
    logger.info(f'Start training model {model.model_path}')
    if valid_data is None:
        train_data, valid_data = train_test_split(train_data, test_size=data_cnf.get('valid', 0.2),
                                                  random_state=random_state)
    train_loader = DataLoader(TFBindDataset(train_data, data_cnf['genome_fasta_file'], data_cnf['bigwig_file'], **model_cnf['padding']),
                              batch_size=model_cnf['train']['batch_size'], shuffle=False)
    valid_loader = DataLoader(TFBindDataset(valid_data, data_cnf['genome_fasta_file'], data_cnf['bigwig_file'], **model_cnf['padding']),
                              batch_size=model_cnf['valid']['batch_size'])
    model.train(train_loader, valid_loader, class_weights_dict, **model_cnf['train'])
    logger.info(f'Finish training model {model.model_path}')


def test(model, data_cnf, model_cnf, test_data, islist=False, bigwig=None):
    if islist:
        data_loader = DataLoader(TFBindDataset(test_data, data_cnf['genome_fasta_file'], bigwig, **model_cnf['padding']),
                             batch_size=model_cnf['test']['batch_size'])
    else:
        data_loader = DataLoader(TFBindDataset(test_data, data_cnf['genome_fasta_file'], data_cnf['bigwig_file'], **model_cnf['padding']),
                             batch_size=model_cnf['test']['batch_size'])
    return model.predict(data_loader)


def generate_cv_id(length, num_groups=5):
    base_size = length // num_groups
    extra_size = length % num_groups
    group_sizes = [base_size + 1 if i < extra_size else base_size for i in range(num_groups)]
    #labels = np.repeat(np.arange(1, num_groups + 1), group_sizes)
    labels = np.concatenate([np.full(size, i) for i, size in enumerate(group_sizes)])
    return labels


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True))
@click.option('-m', '--model-cnf', type=click.Path(exists=True))
@click.option('--mode', type=click.Choice(('train', 'eval', 'eval_list', 'predict', 'predict_list', '5cv', 'loo', 'lomo')), default=None)
@click.option('-s', '--start-id', default=0)
@click.option('-n', '--num_models', default=1)
@click.option('-c', '--continue_train', is_flag=True)
def main(data_cnf, model_cnf, mode, start_id, num_models, continue_train):
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    logger.info(f'check important parameter')
    logger.info(f"Model Name: {model_cnf['name']}, tf number: {len(model_cnf['model']['all_tfs'])}, cutoff: {CUTOFF}")
        
    model_name = model_cnf['name']
    model_path = Path(model_cnf['path'])/f'{model_name}.pt'
    res_path = Path(data_cnf['results'])/f'{model_name}'
    Path(data_cnf['results']).mkdir(parents=True, exist_ok=True)
    model_cnf.setdefault('ensemble', 20)
    get_data_fn = partial(get_data_lazy, genome_fasta_file= data_cnf['genome_fasta_file'])
    all_tfs = model_cnf['model']['all_tfs']

    model_structure = model_cnf['model_structure']
    if model_structure == 'DanQ':
        from tfnet.networks_danq import Danq as Selected_Model
    elif model_structure == 'scFAN':
        from tfnet.networks_scfan import scFAN as Selected_Model
    elif model_structure == 'DeepATT':
        from tfnet.networks_deepatt import DeepATT as Selected_Model
    elif model_structure == 'DeepFormer':
        from tfnet.networks_deepformer import DeepFormer as Selected_Model
    elif model_structure == 'TFNet3':
        from tfnet.networks_tfnet3 import TFNet3 as Selected_Model
    else:
        raise ValueError(f"Unknown network type: {model_structure}")

    classweights = model_cnf['classweights']

    if classweights and ( mode != "eval" and mode != "eval_list" and mode != "predict" and mode != "predict_list"):
        class_weights_dict = calculate_class_weights_dict(data_cnf['train'])
    else :
        class_weights_dict = None

    if mode == "train":
        for model_id in range(start_id, start_id + num_models):
            if continue_train:
                logger.info(f'Continue train Mode')
                model = Model(Selected_Model, model_path=model_path.with_stem(f'{model_path.stem}-{model_id}'), class_weights_dict = class_weights_dict,
                          **model_cnf['model'])
                logger.info(f'Loading Model: {model_path.stem}-{model_id}')
                model.load_model()

                train_data = get_data_fn(data_cnf['train']) if mode is None or mode == 'train' else None
                valid_data = get_data_fn(data_cnf['valid']) if train_data is not None and 'valid' in data_cnf else None

                train(model, data_cnf, model_cnf, train_data=train_data, valid_data=valid_data, class_weights_dict = class_weights_dict)
            else:
                if not model_path.with_stem(f'{model_path.stem}-{model_id}').exists():
                    model = Model(Selected_Model, model_path=model_path.with_stem(f'{model_path.stem}-{model_id}'), class_weights_dict = class_weights_dict,
                          **model_cnf['model'])
                    
                    train_data = get_data_fn(data_cnf['train']) if mode is None or mode == 'train' else None
                    valid_data = get_data_fn(data_cnf['valid']) if train_data is not None and 'valid' in data_cnf else None

                    train(model, data_cnf, model_cnf, train_data=train_data, valid_data=valid_data, class_weights_dict = class_weights_dict)
                else:
                    logger.info(f'Model already exsit: {model_path.stem}-{model_id}')                    
            
    elif mode == 'eval':
        test_data = get_data_fn(data_cnf['test'])
        chr, start, stop, targets_lists = [x[0] for x in test_data], [x[1] for x in test_data], [x[2] for x in test_data], [x[-1] for x in test_data]

        scores_lists = []
        for model_id in range(start_id, start_id + num_models):
            model = Model(Selected_Model, model_path=model_path.with_stem(f'{model_path.stem}-{model_id}'), class_weights_dict = class_weights_dict,
                          **model_cnf['model'])
            scores_lists.append(test(model, data_cnf, model_cnf, test_data=test_data))
        output_eval(chr, start, stop, np.array(targets_lists), np.mean(scores_lists, axis=0), all_tfs, res_path)
        
    
    elif mode == 'eval_list':
        for index, file_path in enumerate(data_cnf['test_list']):
            test_data = get_data_fn(data_cnf['test_list'][index])
            test_prefix = Path(data_cnf['test_list'][index]).stem
            eval_path = Path(data_cnf['results'])/f'{model_name}.{test_prefix}'

            chr, start, stop, targets_lists = [x[0] for x in test_data], [x[1] for x in test_data], [x[2] for x in test_data], [x[-1] for x in test_data]

            scores_lists = []
            for model_id in range(start_id, start_id + num_models):
                model = Model(Selected_Model, model_path=model_path.with_stem(f'{model_path.stem}-{model_id}'), class_weights_dict = class_weights_dict,
                            **model_cnf['model'])
                #pdb.set_trace()
                scores_lists.append(test(model, data_cnf, model_cnf, test_data=test_data, islist=True, bigwig = data_cnf['bigwig_file_list_test'][index]))
            output_eval(chr, start, stop, np.array(targets_lists), np.mean(scores_lists, axis=0), all_tfs, eval_path)    

    
    elif mode == 'predict':
        predict_data = get_data_fn(data_cnf['predict'])
        predict_prefix = Path(data_cnf['predict']).stem
        predict_path = Path(data_cnf['results'])/f'{model_name}.{predict_prefix}'              
        chr, start, stop, targets_lists = [x[0] for x in predict_data], [x[1] for x in predict_data], [x[2] for x in predict_data], [x[-1] for x in predict_data]

        scores_lists = []
        for model_id in range(start_id, start_id + num_models):
            model = Model(Selected_Model, model_path=model_path.with_stem(f'{model_path.stem}-{model_id}'), class_weights_dict = class_weights_dict,
                          **model_cnf['model'])
            scores_lists.append(test(model, data_cnf, model_cnf, test_data=predict_data))
        output_predict(chr, start, stop, np.mean(scores_lists, axis=0), predict_path)


    elif mode == 'predict_list':
        for index, file_path in enumerate(data_cnf['predict_list']):            
            predict_data = get_data_fn(data_cnf['predict_list'][index])
            predict_prefix = Path(data_cnf['predict_list'][index]).stem
            predict_path = Path(data_cnf['results'])/f'{model_name}.{predict_prefix}'              
            chr, start, stop, targets_lists = [x[0] for x in predict_data], [x[1] for x in predict_data], [x[2] for x in predict_data], [x[-1] for x in predict_data]

            scores_lists = []
            for model_id in range(start_id, start_id + num_models):
                model = Model(Selected_Model, model_path=model_path.with_stem(f'{model_path.stem}-{model_id}'), class_weights_dict = class_weights_dict,
                            **model_cnf['model'])
                scores_lists.append(test(model, data_cnf, model_cnf, test_data=predict_data, islist=True, bigwig = data_cnf['bigwig_file_list_predict'][index]))
            output_predict(chr, start, stop, np.mean(scores_lists, axis=0), predict_path)
        

    elif mode == '5cv':
        data = np.asarray(get_data_fn(data_cnf['train']), dtype=object)
        data_group_name, atac_signal, data_truth = [x[0] for x in data], [x[1] for x in data], [x[2] for x in data]
        cv_id_len = data.shape[0]
        # ---------------------- generate cv id for use rather than read a input ---------------------- #
        cv_id = generate_cv_id(cv_id_len)
        assert len(data) == len(cv_id)
        scores_list = []
        for model_id in range(start_id, start_id + num_models):
            scores_ = np.empty(len(data)*len(all_tfs), dtype=np.float32).reshape(len(data), len(all_tfs))
            for cv_ in range(5):
                
                train_data, test_data = data[cv_id != cv_], data[cv_id == cv_]
                model = Model(Selected_Model, model_path=model_path.with_stem(f'{model_path.stem}-{model_id}-CV{cv_}'), class_weights_dict = class_weights_dict,
                              **model_cnf['model'])
                if not continue_train or not model.model_path.exists():
                    train(model, data_cnf, model_cnf, train_data=train_data, class_weights_dict = class_weights_dict)
                scores_[cv_id == cv_] = test(model, model_cnf, test_data=test_data)

                scores_list.append(scores_)
                #pdb.set_trace()

                #output_res(np.array(data_group_name)[cv_id == cv_], np.array(data_truth)[cv_id == cv_], np.mean(scores_[cv_id == cv_], axis=0),
                output_eval(np.array(data_group_name)[cv_id == cv_], np.array(data_truth)[cv_id == cv_], scores_[cv_id == cv_], all_tfs,          
                       res_path.with_name(f'{res_path.stem}-5CV'))


    elif mode == 'loo' or mode == 'lomo':
        data = np.asarray(get_data_fn(data_cnf['train']), dtype=object)
        with open(data_cnf['cv_id']) as fp:
            cv_id = np.asarray([int(line) for line in fp])
        scores_list = []
        for model_id in range(start_id, start_id + num_models):
            group_names, group_names_, truth_, scores_ = np.asarray([x[0] for x in data]), [], [], []
            for name_ in sorted(set(group_names)):
                train_data, train_cv_id = data[group_names != name_], cv_id[group_names != name_]
                test_data, test_cv_id = data[group_names == name_], cv_id[group_names == name_]
                if len(test_data) > 30 and len([x[-1] for x in test_data if x[-1] >= CUTOFF]) >= 3:
                    for cv_ in range(5):
                        model = Model(Selected_Model,
                                      model_path=model_path.with_stem(F'{model_path.stem}-{name_}-{model_id}-CV{cv_}'), class_weights_dict = class_weights_dict,
                                      **model_cnf['model'])
                        if not model.model_path.exists() or not continue_train:
                            train(model, data_cnf, model_cnf, train_data[train_cv_id != cv_], class_weights_dict=class_weights_dict)
                        test_data_ = test_data[test_cv_id == cv_]
                        group_names_ += [x[0] for x in test_data_]
                        truth_ += [x[-1] for x in test_data_]
                        scores_ += test(model, model_cnf, test_data_).tolist()
            scores_list.append(scores_)
            output_eval(group_names_, truth_, np.mean(scores_list, axis=0), all_tfs, res_path.with_name(f'{res_path.stem}-LOMO'))


if __name__ == '__main__':
    main()
