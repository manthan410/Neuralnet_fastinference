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
from torch.autograd import Function

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
  
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
def sanatize_onnx(model):
    """ONNX does not support binary layers out of the box and exporting custom layers is sometimes difficult. This function sanatizes a given MLP so that it can be exported into an onnx file. To do so, it replaces all BinaryLinear layer with regular nn.Linear layers and BinaryTanh with Sign() layers. Weights and biases are copied and binarized as required.

    Args:
        model: The pytorch model.

    Returns:
        Model: The pytorch model in which each binary layer is replaced with the appropriate float layer.
    """

    # Usually I would use https://pytorch.org/docs/stable/generated/torch.heaviside.html for exporting here, but this is not yet supported in ONNX files. 
    class Sign(nn.Module):
        def forward(self, input):
            return torch.where(input > 0, torch.tensor([1.0]), torch.tensor([-1.0]))
            # return torch.sign(input)

    for name, m in model._modules.items():
        print("Checking {}".format(name))

        if isinstance(m, BinaryLinear):
            print("Replacing {}".format(name))
            # layer_old = m
            layer_new = nn.Linear(m.in_features, m.out_features, hasattr(m, 'bias'))
            if (hasattr(m, 'bias')):
                layer_new.bias.data = binarize(m.bias.data)
            layer_new.weight.data = binarize(m.weight.data)
            model._modules[name] = layer_new

        if isinstance(m, BinaryTanh):
            model._modules[name] = Sign()

        if isinstance(m, BinaryConv2d):
            print("Replacing {}".format(name))
            # layer_old = m
            layer_new = nn.Conv2d(
                in_channels = m.in_channels, 
                out_channels = m.out_channels, 
                kernel_size = m.kernel_size, 
                stride = m.stride, 
                #padding = m.padding,
                bias = hasattr(m, 'bias')
            )

            if (hasattr(m, 'bias')):
                layer_new.bias.data = binarize(m.bias.data)
            layer_new.weight.data = binarize(m.weight.data)
            model._modules[name] = layer_new
        
        # if isinstance(m, nn.BatchNorm2d):
        #     layer_new = WrappedBatchNorm(m)
        #     model._modules[name] = layer_new

    return model

class BinarizeF(Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input > 0] = 1
        output[input <= 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        #return grad_output, None
        grad_input = grad_output.clone()
        return grad_input#, None

# aliases
binarize = BinarizeF.apply

class BinaryConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(BinaryConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            binary_weight = binarize(self.weight)

            return F.linear(input, binary_weight)
        else:
            binary_weight = binarize(self.weight)
            binary_bias = binarize(self.bias)
            return F.conv2d(input, binary_weight, binary_bias)

class BinaryLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(BinaryLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            binary_weight = binarize(self.weight)

            return F.linear(input, binary_weight)
        else:
            binary_weight = binarize(self.weight)
            binary_bias = binarize(self.bias)
            return F.linear(input, binary_weight, binary_bias)

class BinaryTanh(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Hardtanh(*args, **kwargs)

    def forward(self, input):
        output = self.hardtanh(input)
        output = binarize(output)
        return output

class VGG(pl.LightningModule):

    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.infl_ratio=1
        self.features = nn.Sequential(
            BinaryConv2d(1, 16*self.infl_ratio, kernel_size=3, stride=1, padding=1,
                      bias=True),
            nn.BatchNorm2d(16*self.infl_ratio),
            BinaryTanh(inplace=True),

            BinaryConv2d(16*self.infl_ratio, 16*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(16*self.infl_ratio),
            BinaryTanh(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),


            BinaryConv2d(16*self.infl_ratio, 32*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32*self.infl_ratio),
            BinaryTanh(inplace=True),


            BinaryConv2d(32*self.infl_ratio, 32*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32*self.infl_ratio),
            BinaryTanh(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            Flatten(),
            BinaryLinear(32 * 4 * 4, 1024, bias=True),
            nn.BatchNorm1d(1024),
            BinaryTanh(inplace=True),
            #nn.Dropout(0.5),
            BinaryLinear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            BinaryTanh(inplace=True),
            #nn.Dropout(0.5),
            BinaryLinear(1024, num_classes, bias=True),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view((batch_size, 1, 28, 28))

        x = self.features(x)
        #x = x.view(batch_size, -1)
        #x = x.view(-1, 32 * 4 * 4)
        x = self.classifier(x)
        #x = torch.log_softmax(x, dim=1)
        return x
   

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)
        #return None

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        #
        #_, preds = torch.max(logits, dim=1)
        #acc= torch.sum(preds ==y.data)/ (y.shape[0] * 1.0)
        #self.log('train_acc', acc)
        #
        if loss is not None:
            self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        if loss is not None:
            self.log('val_loss', loss)
    """
    def test_step(self, test_batch, batch_idx):
	    x, y = batch
	    x = x.view(x.size(0), -1)
	    y_hat = self.layers(x)
	    loss = self.ce(y_hat, y)
	    y_hat = torch.argmax(y_hat, dim=1)
	    accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
	    output = dict({
		'test_loss': loss,
		'test_acc': torch.tensor(accuracy),
	    })
	    return output
    """
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict(self, X):
        return self.forward(torch.from_numpy(X).float()).argmax(axis=1)
    def on_epoch_start(self):
        print('\n')

def eval_model(model, train_data ,test_data, out_path, name):
    print("Fitting {}".format(name))

   # x_train_tensor = torch.from_numpy(x_train).float()
   # y_train_tensor = torch.from_numpy(y_train).long()

    #train_dataloader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=64)
    train_dataloader = DataLoader(train_data, batch_size=64, num_workers=5)
    val_loader = None  # DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=64)

    trainer = pl.Trainer(max_epochs=1, default_root_dir=out_path, enable_progress_bar=True)
    trainer.fit(model, train_dataloader, val_loader)
    model.eval()  # This is to make sure that model is removed from the training state to avoid errors
   # preds = model.predict(x_test)

   # accuracy = accuracy_score(y_test, preds) * 100.0

    dummy_x = torch.randn(1,1,28,28, requires_grad=False)
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
    model = sanatize_onnx(model)
    print(model)
    torch.onnx.export(model, dummy_x, onnx_path, training=torch.onnx.TrainingMode.PRESERVE, export_params=True, opset_version=11,
                      do_constant_folding=True, input_names=['input'], output_names=['output'])
    
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
    n_channels =32
    depth=4
    #model = SimpleCNN(n_features, n_classes, args.binarize, args.outpath)

    model =VGG()

    #eval_model(model, x_train, y_train, x_test, y_test, args.outpath, args.modelname)
    eval_model(model,train_data, test_data, args.outpath, args.modelname)
    print("")


if __name__ == '__main__':
    main()
