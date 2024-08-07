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
from typing import Optional, Mapping
from tfnet.evaluation import get_mean_auc, get_mean_f1, get_label_ranking_average_precision_score, get_mean_accuracy_score, get_mean_balanced_accuracy_score, get_mean_recall, get_mean_aupr
import matplotlib.pyplot as plt
import pdb
import warnings 

warnings.filterwarnings("ignore",category=UserWarning)
#mps_device = torch.device("mps")
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# modifity the initial pcc best to -1, due to small dataset

__all__ = ['Model']


# code
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0, init_value_low= float('inf'), init_value_high= 0):
        # init_value = float('inf') for valid loss, init_value = 0 for balanced accuracy
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.init_value_low = init_value_low
        self.init_value_high = init_value_high

    def early_stop_low(self, exam_value):
        if exam_value < self.init_value_low:
            self.init_value_low = exam_value
            self.counter = 0
        elif exam_value > (self.init_value_low + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def early_stop_high(self, exam_value):
        if exam_value > self.init_value_high:
            self.init_value_high = exam_value
            self.counter = 0
        elif exam_value < (self.init_value_high - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

class Model(object):
    """

    """
    def __init__(self, network, model_path, class_weights_dict = None, **kwargs):
        self.model = self.network = network(**kwargs).to(device)

        if class_weights_dict:
            self.model_path =  Path(model_path)
        else:
            self.loss_fn, self.model_path = nn.BCEWithLogitsLoss(), Path(model_path)
        
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.optimizer = None
        self.training_state = {}

        self.early_stopper_1 = EarlyStopper(patience=8, min_delta=0.005)
        self.early_stopper_2 = EarlyStopper(patience=8, min_delta=0.005)

    def get_scores(self, inputs, **kwargs):
        return self.model(inputs.to(device), **kwargs)
    
    def cal_loss(self, scores, targets, class_weights_dict):
        if class_weights_dict:
            weight = torch.zeros(targets.shape)  
            for i in range(targets.shape[0]):
                for j in range(targets.shape[1]):
                    weight[i][j] = class_weights_dict[j][int(targets[i][j])]
            loss = nn.functional.binary_cross_entropy_with_logits(scores, targets.to(device), weight.to(device),reduction='mean')
            
        else:
            loss = self.loss_fn(scores, targets.to(device))
            
        return loss

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor, class_weights_dict= None, **kwargs):
        self.optimizer.zero_grad()
        self.model.train()
        loss = self.cal_loss(self.get_scores(inputs, **kwargs), targets, class_weights_dict)
        loss.backward()
        self.optimizer.step(closure=None)
        return loss.item()

    @torch.no_grad()
    def predict_step(self, inputs: torch.Tensor, **kwargs):
        self.model.eval()
        return self.get_scores(inputs, **kwargs).to(device)

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
            balanced_accuracy,valid_loss = self.valid(valid_loader, verbose, epoch_idx, train_loss)
            if self.early_stopper_1.early_stop_low(valid_loss):
                logger.info(f'Early Stopping due to valid loss')
                break
            if self.early_stopper_2.early_stop_high(balanced_accuracy):
                logger.info(f'Early Stopping due to balanced accuracy')
                break            
        # ---------------------- record loss pcc for each epoch and plot---------------------- #


    def valid(self, valid_loader, verbose, epoch_idx, train_loss, **kwargs):
        scores, targets = self.predict(valid_loader, valid=True, **kwargs), valid_loader.dataset.bind_list
        valid_loss = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(scores).to(device), torch.tensor(targets).to(device))

        mean_auc = get_mean_auc(targets, scores)
        f1_score = get_mean_f1(targets, scores)
        recall_score = get_mean_recall(targets, scores)
        aupr = get_mean_aupr(targets, scores)
        lrap = get_label_ranking_average_precision_score(targets, scores)
        accuracy = get_mean_accuracy_score(targets, scores)
        balanced_accuracy = get_mean_balanced_accuracy_score(targets, scores, axis = 1)

        if mean_auc > self.training_state['best']:
            self.save_model()
            self.training_state['best'] = mean_auc
        if verbose:
            logger.info(f'Epoch: {epoch_idx}  '
                        f'train loss: {train_loss:.5f}  '
                        f'valid loss: {valid_loss:.5f}  ' 
                        f'mean_auc: {mean_auc:.5f}  '
                        f'aupr: {aupr:.5f}  '
                        f'recall score: {recall_score:.5f}  '
                        f'f1 score: {f1_score:.5f}  '
                        f'lrap: {lrap:.5f}  '
                        f'accuracy: {accuracy:.5f}  '
                        f'balanced accuracy: {balanced_accuracy:.5f}'
                        )
            
        # ---------------------- record data for plot ---------------------- #
        with open('results/train_record.txt', 'a') as output_file:
            writer = csv.writer(output_file, delimiter="\t")
            writer.writerow([epoch_idx, train_loss, valid_loss.item(), mean_auc, f1_score, lrap, accuracy, balanced_accuracy])

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
            plt.plot(loss_data[:,4], label='f1')
            plt.legend(loc='best')
            plt.subplot(1, 4, 4)     
            plt.plot(loss_data[:,7], label='balanced accuracy')
            plt.legend(loc='best')
            plt.savefig('results/train.pdf')
            plt.close()

        return balanced_accuracy, valid_loss

    def predict(self, data_loader: DataLoader, valid=False, **kwargs):
        if not valid:
            self.load_model()
        return np.concatenate([torch.sigmoid(self.predict_step(data_x, **kwargs)).cpu()
                               for data_x, _ in tqdm(data_loader, leave=False, dynamic_ncols=True)], axis=0)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        #self.model.load_state_dict(torch.load(self.model_path, map_location=device))
