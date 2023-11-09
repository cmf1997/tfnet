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

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from logzero import logger
from typing import Optional, Mapping, Tuple
from tfnet.evaluation import get_auc, get_pcc, get_f1, get_label_ranking_average_precision_score

mps_device = torch.device("mps")

# modifity the initial pcc best to -1, due to small dataset

__all__ = ['Model']


# code
class Model(object):
    """

    """
    def __init__(self, network, model_path, **kwargs):
        self.model = self.network = network(**kwargs).to(mps_device)
        # consider Cross Entropy as loss_fn
        #self.loss_fn, self.model_path = nn.CrossEntropyLoss(), Path(model_path)

        self.loss_fn, self.model_path = nn.BCELoss(), Path(model_path)
        
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.optimizer = None
        self.training_state = {}

    def get_scores(self, inputs, **kwargs):
        return self.model(*(x.to(mps_device) for x in inputs), **kwargs)

    def loss_and_backward(self, scores, targets):
        loss = self.loss_fn(scores, targets.to(mps_device))
        loss.backward()
        return loss

    def train_step(self, inputs: Tuple[torch.Tensor, torch.Tensor], targets: torch.Tensor, **kwargs):
        self.optimizer.zero_grad()
        self.model.train()
        loss = self.loss_and_backward(self.get_scores(inputs, **kwargs), targets)
        self.optimizer.step(closure=None)
        return loss.item()

    @torch.no_grad()
    def predict_step(self, inputs: Tuple[torch.Tensor, torch.Tensor], **kwargs):
        self.model.eval()
        return self.get_scores(inputs, **kwargs).cpu()

    def get_optimizer(self, optimizer_cls='Adadelta', weight_decay=1e-3, **kwargs):
        if isinstance(optimizer_cls, str):
            optimizer_cls = getattr(torch.optim, optimizer_cls)
        self.optimizer = optimizer_cls(self.model.parameters(), weight_decay=weight_decay, **kwargs)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, opt_params: Optional[Mapping] = (),
              num_epochs=20, verbose=True, **kwargs):
        self.get_optimizer(**dict(opt_params))
        self.training_state['best'] = -1
        for epoch_idx in range(num_epochs):
            train_loss = 0.0
            for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch_idx}', leave=False, dynamic_ncols=True):
                train_loss += self.train_step(inputs, targets, **kwargs) * len(targets)
            train_loss /= len(train_loader.dataset)
            self.valid(valid_loader, verbose, epoch_idx, train_loss)
        # ---------------------- record loss pcc for each epoch and plot---------------------- #



        


    def valid(self, valid_loader, verbose, epoch_idx, train_loss, **kwargs):
        scores, targets = self.predict(valid_loader, valid=True, **kwargs), valid_loader.dataset.targets
        #print("valid scores shape",scores.shape, "valid targets shape", targets.shape)
        #auc, pcc = get_auc(targets, scores), get_pcc(targets, scores)

        #auc = get_auc(targets, scores)
        #print(scores)
        pcc = get_pcc(targets, scores)
        f1_score = get_f1(targets, scores)
        lrap = get_label_ranking_average_precision_score(targets, scores)


        if pcc > self.training_state['best']:
            self.save_model()
            self.training_state['best'] = pcc
        if verbose:
            logger.info(f'Epoch: {epoch_idx}  '
                        f'train loss: {train_loss:.5f}  '
                        f'f1 score: {f1_score:.3f}  '
                        f'PCC: {pcc:.3f}  '
                        f'lrap: {lrap:.3f}  '
                        )

    def predict(self, data_loader: DataLoader, valid=False, **kwargs):
        if not valid:
            self.load_model()
        return np.concatenate([self.predict_step(data_x, **kwargs)
                               for data_x, _ in tqdm(data_loader, leave=False, dynamic_ncols=True)], axis=0)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
