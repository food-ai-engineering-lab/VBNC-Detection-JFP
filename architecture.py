import os
import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import MSELoss
from torch.optim import lr_scheduler
import torch.nn.functional as F
import pytorch_lightning as pl
import timm

# define the model class
class MicrocolonyNet(pl.LightningModule):
    
    # initialization function
    def __init__(self):
        super(MicrocolonyNet, self).__init__()

        # create an instance of the EfficientNet model with 100 output classes, pretrained weights
        self.model = timm.create_model('efficientnetv2_rw_s', num_classes = 100, pretrained=True)
        
    # forward pass function
    def forward(self,x):
        out = self.model(x) # pass the input through the model
        return out
        
    # training step function
    def training_step(self, batch, batch_idx):
        input = batch[0]
        target = batch[1]
        pred = self.model(input)
        # compute the loss using cross entropy loss function
        loss = F.cross_entropy(pred.float(), target)
        # log the training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)        
        return loss

    # validation step function
    def validation_step(self, batch, batch_idx):
        input = batch[0]
        target = batch[1]
        pred = self.model(input)
        # compute the loss using cross entropy loss function
        loss = F.cross_entropy(pred.float(), target)
        # log the validation loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    # testing step function
    def test_step(self, batch, batch_idx):
        input = batch[0]
        target = batch[1]
        pred = self.model(input)
        # compute the loss using cross entropy loss function
        loss = F.cross_entropy(pred.float(), target)
        # log the test loss
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # define the optimizer and learning rate scheduler
    def configure_optimizers(self):
        # create an instance of the AdamW optimizer
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2) # default: lr=1e-3
        # sch = lr_scheduler.StepLR(opt, step_size=10, gamma=0.3) # every epoch by default
        # return ({'optimizer': opt, 'lr_scheduler':sch})
        
        # create a learning rate scheduler that decreases the learning rate every 10 epochs by a factor of 0.3
        sch = {'scheduler': lr_scheduler.StepLR(opt, step_size=10, gamma=0.3)}
        return [opt], [sch]
