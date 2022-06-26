#!/usr/bin/env python3

import os
import sys
import pickle
import tarfile
from datetime import datetime
from functools import partial
import argparse
import glob

import numpy as np
import pandas as pd

import scipy
import onnx
import torch.onnx
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from torch.autograd import Variable
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
from torchsummaryX import summary

import torch
#import torch.nn as nn
from torch import nn
#import torch.nn.functional as F
from torch.nn import functional as F
import math
from collections import OrderedDict
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
#from pysembles.Utils import Flatten
from adabelief_pytorch import AdaBelief
from BinarisedNeuralNetworks import BinaryModel


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class VGGNet(pl.LightningModule):
    def __init__(self, input_size = (1,28,28), n_channels = 32, depth = 4, hidden_size = 512, p_dropout = 0.0, n_classes = 10):
        super().__init__()

        in_channels = input_size[0]
        
        def make_layers(level, n_channels):
            return [
                nn.Conv2d(in_channels if level == 0 else level*n_channels, (level+1)*n_channels, kernel_size=3, padding=1, stride = 1, bias=True),
                nn.BatchNorm2d((level+1)*n_channels),
                nn.ReLU(),
                nn.Conv2d((level+1)*n_channels, (level+1)*n_channels, kernel_size=3, padding=1, stride = 1, bias=True),
                nn.BatchNorm2d((level+1)*n_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2)
            ]

        model = []
        for i in range(depth):
            model.extend(make_layers(i, n_channels))

        # This is kinda hacking but it works well to automatically determine the size of the linear layer
        x = torch.rand((1,*input_size)).type(torch.FloatTensor)
        for l in model:
            x = l(x)
        lin_size = x.view(1,-1).size()[1]
        print("lin_size",lin_size)

        model.extend(
            [
                Flatten(),
                nn.Linear(lin_size, hidden_size),
                nn.Dropout(p_dropout) if p_dropout > 0 else None,
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_classes)
            ]
        )

        model = filter(None, model)
        self.layers_ = nn.Sequential(*model)
        

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view((batch_size, 1, 28, 28))
        return self.layers_(x)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)
        #return None

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        preds = self.layers_(x)
        acc = (preds.argmax(dim=-1) == y).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        if loss is not None:
            self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        preds = self.layers_(x).argmax(dim=-1)
        acc = (y == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)
        if loss is not None:
            self.log('val_loss', loss)

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        #optimizer= torch.optim.SGD(self.parameters(),lr=0.1, momentum=0.9, nesterov= True, weight_decay=1e-4)
        optimizer = AdaBelief(self.parameters(), lr=1e-2 ,eps=1e-12, betas= (0.9,0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        return [optimizer], [scheduler]
        #return optimizer
    def test_step(self, test_batch, batch_idx):
        imgs, labels = test_batch
        preds = self.layers_(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)


    def predict(self, X):
        return self.forward(torch.from_numpy(X).float()).argmax(axis=1)
    def on_epoch_start(self):
        print('\n')

def eval_model(model, train_data ,test_data, out_path, name):
    print("Fitting {}".format(name))

   # x_train_tensor = torch.from_numpy(x_train).float()
   # y_train_tensor = torch.from_numpy(y_train).long()

    #train_dataloader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=64)
    train_dataloader = DataLoader(train_data, batch_size=256, num_workers=15)
    val_loader = None  # DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=256, num_workers=15)
    trainer = pl.Trainer(max_epochs=150, default_root_dir=out_path, enable_progress_bar=True)
    trainer.fit(model, train_dataloader, val_loader)
    test_result= trainer.test(model, dataloaders=test_dataloader)
    result = {"test": test_result[0]["test_acc"]}
    print("result", result)

   # trainer.test(dataloaders = test_dataloaders)
    model.eval()  # This is to make sure that model is removed from the training state to avoid errors
   # preds = model.predict(x_test)

   # accuracy = accuracy_score(y_test, preds) * 100.0

    dummy_x = torch.randn(1,784, requires_grad=False)
    '''
    djson = {
        "accuracy": accuracy,
        "name": name,
        #"batch-latency": batch_time / x_test.shape[0],
        #"single-latency": single_time / x_test.shape[0]
    }
    print("accuracy: {}".format(djson["accuracy"]))'''
    # print("batch-latency: {}".format(djson["batch-latency"]))
    # print("single-latency: {}".format(djson["single-latency"]))

    #with open(os.path.join(out_path, name + ".json"), "w") as outfile:
    #    json.dump(djson, outfile)  # , cls=NumpyEncoder

   # if not (name.endswith("ONNX") or name.endswith("onnx")):
   #     name += ".onnx"

    onnx_path = os.path.join(out_path, name + ".onnx")
    print("Exporting {} to {}".format(name, onnx_path))

    #print("Exporting {} to {}".format(name, out_path))
    # Export the model, onnx file name:super_resolution.onnx
    print(model)
    torch.onnx.export(model, dummy_x, onnx_path, training=torch.onnx.TrainingMode.PRESERVE, export_params=True, opset_version=12,
                      do_constant_folding=True, input_names=['input'], output_names=['output'],dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})

    #onnx_model = onnx.load(os.path.join(out_path, name))
    #onnx.checker.check_model(onnx_model)


def main():
    parser = argparse.ArgumentParser(description='Benchmark various CNN optimizations on the MNIST / Fashion dataset.')
    parser.add_argument('--outpath', required=True, help='Folder where data should written to.')
    parser.add_argument('--modelname', required=False, default="model", help='Modelname')
    #parser.add_argument('--split','-s', required=False, default=0.2, type=float, help='Test/Train split.')
    #parser.add_argument('--dataset','-d', required=True, help='Dataset to to be downloaded and used. Currently supported are {mnist, fashion}.')
    parser.add_argument("--binarize", "-b", required=False, action='store_true', help="Trains a binarized neural network if true.")
    args = parser.parse_args()

    def train_transformation():
        return transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def test_transformation():
        return transforms.Compose([
            transforms.ToTensor(),
        ])

    train_data = torchvision.datasets.FashionMNIST(os.path.join(args.outpath + "/data"), train=True, transform= train_transformation(), download =True)
    test_data = torchvision.datasets.FashionMNIST(os.path.join(args.outpath + "/data"), train=False, transform= test_transformation(), download =True)

    #x_train, y_train, x_test, y_test = train_test_split(train_data, test_data, test_size=0.3, random_state=42)

    n_classes = 10
    n_channels =16
    depth=2
    #model = SimpleCNN(n_features, n_classes, args.binarize, args.outpath)
    if args.binarize:
       model = BinaryModel(VGGNet(input_size = (1,28,28), n_channels=n_channels, depth= depth, hidden_size = 1024, p_dropout = 0.0, n_classes=n_classes), keep_activation=True)
    else:
       model = VGGNet(input_size = (1,28,28), n_channels=n_channels, depth=depth,hidden_size = 1024, p_dropout = 0.0, n_classes=n_classes)

    #eval_model(model, x_train, y_train, x_test, y_test, args.outpath, args.modelname)
    eval_model(model,train_data, test_data, args.outpath, args.modelname)
    print("")


if __name__ == '__main__':
    main()
