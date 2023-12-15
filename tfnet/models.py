#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : models.py
@Time : 2023/11/09 11:21:25
@Author : Cmf
@Version : 1.0
@Desc : None
'''

# here put the import lib
import numpy as np
import torch
import torch.nn as nn
import csv

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from logzero import logger
from typing import Optional, Mapping, Tuple
from tfnet.evaluation import get_mean_auc, get_mean_f1, get_label_ranking_average_precision_score, get_mean_accuracy_score, get_mean_balanced_accuracy_score, get_mean_pcc
from tfnet.all_tfs import all_tfs
import matplotlib.pyplot as plt
import pdb

mps_device = torch.device("mps")

# modifity the initial pcc best to -1, due to small dataset

__all__ = ['Model']


# code
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

class Model(object):
    """

    """
    def __init__(self, network, model_path, class_weights_dict = None, **kwargs):
        self.model = self.network = network(**kwargs).to(mps_device)
        # consider Cross Entropy as loss_fn
        #self.loss_fn, self.model_path = nn.CrossEntropyLoss(), Path(model_path)
        if class_weights_dict:


            self.model_path =  Path(model_path)
        else:
            self.loss_fn, self.model_path = nn.BCELoss(), Path(model_path)
        
        
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.optimizer = None
        self.training_state = {}

        self.early_stopper_1 = EarlyStopper(patience=5, min_delta=0.02)
        self.early_stopper_2 = EarlyStopper(patience=5, min_delta=0.05)

    def get_scores(self, inputs, **kwargs):
        return self.model(*(x.to(mps_device) for x in inputs), **kwargs)

    def cal_loss(self, scores, targets, class_weights_dict):
        if class_weights_dict:
            weight = torch.zeros(targets.shape)  
            for i in range(targets.shape[0]):
                for j in range(targets.shape[1]):
                    weight[i][j] = class_weights_dict[j][int(targets[i][j])]
            #loss = nn.functional.binary_cross_entropy(scores, targets.to(mps_device), weight.to(mps_device),reduction='mean')
            loss = nn.functional.binary_cross_entropy_with_logits(scores, targets.to(mps_device), weight.to(mps_device),reduction='mean')
            
        else:
            loss = self.loss_fn(scores, targets.to(mps_device))
            
        return loss

    def train_step(self, inputs: Tuple[torch.Tensor, torch.Tensor], targets: torch.Tensor, class_weights_dict= None, **kwargs):
        self.optimizer.zero_grad()
        self.model.train()
        loss = self.cal_loss(self.get_scores(inputs, **kwargs), targets, class_weights_dict)
        loss.backward()
        self.optimizer.step(closure=None)
        return loss.item()

    @torch.no_grad()
    def predict_step(self, inputs: Tuple[torch.Tensor, torch.Tensor], **kwargs):
        self.model.eval()
        return self.get_scores(inputs, **kwargs).to(mps_device)

    def get_optimizer(self, optimizer_cls='Adadelta', weight_decay=0, betas=None, **kwargs):
        if isinstance(optimizer_cls, str):
            optimizer_cls = getattr(torch.optim, optimizer_cls)
        self.optimizer = optimizer_cls(self.model.parameters(), weight_decay=weight_decay, betas = (0.95,0.9995),**kwargs)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, class_weights_dict=None, opt_params: Optional[Mapping] = (),
              num_epochs=20, verbose=True, **kwargs):
        self.get_optimizer(**dict(opt_params))
        self.training_state['best'] = 0
        for epoch_idx in range(num_epochs):
            train_loss = 0.0
            for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch_idx}', leave=False, dynamic_ncols=True):
                train_loss += self.train_step(inputs, targets, class_weights_dict, **kwargs) * targets.shape[0]
            train_loss /= len(train_loader.dataset)
            balanced_accuracy,valid_loss = self.valid(valid_loader, verbose, epoch_idx, train_loss, class_weights_dict)
            if self.early_stopper_1.early_stop(valid_loss):
                logger.info(f'Early Stopping due to valid loss')
                break
            if self.early_stopper_2.early_stop(balanced_accuracy):
                logger.info(f'Early Stopping due to balanced accuracy')
                break            
        # ---------------------- record loss pcc for each epoch and plot---------------------- #


    def valid(self, valid_loader, verbose, epoch_idx, train_loss, class_weights_dict=None, **kwargs):
        scores, targets = self.predict(valid_loader, valid=True, **kwargs), valid_loader.dataset.targets
        if class_weights_dict:
            valid_loss = self.cal_loss(torch.tensor(scores).to(mps_device), torch.tensor(targets), class_weights_dict)
        else:
            valid_loss = self.loss_fn(torch.tensor(scores).to(mps_device), torch.tensor(targets).to(mps_device))

        #print("valid scores shape",scores.shape, "valid targets shape", targets.shape)
        #scores = nn.functional.sigmoid(torch.tensor(scores))
        mean_auc = get_mean_auc(targets, scores)
        f1_score = get_mean_f1(targets, scores)
        lrap = get_label_ranking_average_precision_score(targets, scores)
        accuracy = get_mean_accuracy_score(targets, scores)
        balanced_accuracy = get_mean_balanced_accuracy_score(targets, scores)
        pcc = get_mean_pcc(targets, scores)

        '''     
        valid_loss = 0 
        for inputs, targets in tqdm(valid_loader, desc=f'valid ', leave=False, dynamic_ncols=True):
            valid_loss += self.cal_loss(self.get_scores(inputs, **kwargs), targets, class_weights_dict).item() * len(targets)
        valid_loss /= len(valid_loader.dataset)
        '''

        

        if balanced_accuracy > self.training_state['best']:
            self.save_model()
            self.training_state['best'] = balanced_accuracy
        if verbose:
            logger.info(f'Epoch: {epoch_idx}  '
                        f'train loss: {train_loss:.5f}  '
                        f'valid loss: {valid_loss:.5f}  ' 
                        f'mean_auc: {mean_auc:.5f}  '
                        f'pcc: {pcc:.5f}  '
                        f'f1 score: {f1_score:.5f}  '
                        f'lrap: {lrap:.5f}  '
                        f'accuracy: {accuracy:.5f}  '
                        f'balanced accuracy: {balanced_accuracy:.5f}'
                        )
            
        # ---------------------- record data for plot ---------------------- #
        with open('results/train_record.txt', 'a') as output_file:
            writer = csv.writer(output_file, delimiter="\t")
            writer.writerow([epoch_idx, train_loss, valid_loss.item(), mean_auc, pcc, f1_score, lrap, accuracy, balanced_accuracy])

        loss_data = np.loadtxt('results/train_record.txt')
        if len(loss_data.shape) != 1:
            f = plt.figure() 
            f.set_figwidth(18) 
            f.set_figheight(4) 
            plt.subplot(1, 4, 1)
            plt.plot(loss_data[:,1], label='train_loss')
            plt.plot(loss_data[:,2], label='valid_loss')
            plt.legend(loc='best')
            plt.subplot(1, 4, 2)
            plt.plot(loss_data[:,3], label='mean_auc')
            plt.legend(loc='best')
            plt.subplot(1, 4, 3)           
            plt.plot(loss_data[:,5], label='f1')
            plt.legend(loc='best')
            plt.subplot(1, 4, 4)     
            plt.plot(loss_data[:,8], label='balanced accuracy')
            plt.legend(loc='best')
            plt.savefig('results/train.pdf')
            plt.close()

        return balanced_accuracy, valid_loss

    def predict(self, data_loader: DataLoader, valid=False, **kwargs):
        if not valid:
            self.load_model()
        return np.concatenate([nn.functional.sigmoid(self.predict_step(data_x, **kwargs)).cpu()
                               for data_x, _ in tqdm(data_loader, leave=False, dynamic_ncols=True)], axis=0)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
