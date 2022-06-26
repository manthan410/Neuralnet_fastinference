'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
#import torch
#import torch.nn as nn
#import torch.nn.functional as F

import sys
import pandas as pd
import numpy as np
import os
import argparse

import json
import torch.onnx
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.autograd import Function

from test_utils import test_implementations

def sanatize_onnx(model):
    return model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        ConvLayer = nn.Conv2d
        LinearLayer = nn.Linear
        Activation = nn.ReLU

        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.a1 = Activation()

        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.a2 = Activation()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                ConvLayer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.a1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.a2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()

        ConvLayer = nn.Conv2d
        LinearLayer = nn.Linear
        Activation = nn.ReLU

        self.conv1 = ConvLayer(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.a1 = Activation()

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = ConvLayer(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.a2 = Activation()
        self.a3 = Activation()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                ConvLayer(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.a1(self.bn1(self.conv1(x)))
        out = self.a2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.a3(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, outpath='.'):
        super(ResNet, self).__init__()

        ConvLayer = nn.Conv2d
        LinearLayer = nn.Linear
        Activation = nn.ReLU

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # FROM 3 TO 1
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 2048 => 16
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 1024
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 512
        self.a1 = Activation()
        self.linear = LinearLayer(512 * block.expansion, num_classes)

        self.outpath = outpath

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.a1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)
        #return None

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        if loss is not None:
            self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        if loss is not None:
            self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict(self, X):
        return self.forward(torch.from_numpy(X).float()).argmax(axis=1)   
        
    def on_epoch_start(self):
        print('\n')

    def fit(self, X, y):
        XTrainT = torch.from_numpy(X).float()
        YTrainT = torch.from_numpy(y).long()

        train_dataloader = DataLoader(TensorDataset(XTrainT, YTrainT), batch_size=64)
        val_loader = None 

        trainer = pl.Trainer(max_epochs = 1, default_root_dir = self.outpath, progress_bar_refresh_rate = 0)
        trainer.fit(self, train_dataloader, val_loader)
        self.eval()

    def store(self, out_path, accuracy, model_name):
        dummy_x = torch.randn(1, self.input_dim, requires_grad=False)

        djson = {
            "accuracy":accuracy,
            "name":model_name
        }

        with open(os.path.join(out_path, model_name + ".json"), "w") as outfile:  
            json.dump(djson, outfile) #, cls=NumpyEncoder

        onnx_path = os.path.join(out_path,model_name+".onnx")
        print("Exporting {} to {}".format(model_name,onnx_path))
        model = sanatize_onnx(self)
        # https://github.com/pytorch/pytorch/issues/49229
        # set torch.onnx.TrainingMode.PRESERVE
        torch.onnx.export(model, dummy_x,onnx_path, training=torch.onnx.TrainingMode.PRESERVE, export_params=True, opset_version=11, do_constant_folding=True, input_names = ['input'],  output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
        
        return onnx_path

def ResNet11():
    return ResNet(BasicBlock, [1, 1, 1, 1])


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

def main():
    parser = argparse.ArgumentParser(description='Benchmark various CNN optimizations on the MNIST / Fashion dataset.')
    parser.add_argument('--outpath', required=True, help='Folder where data should written to.')
    parser.add_argument('--modelname', required=False, default="model", help='Modelname')
    parser.add_argument('--split','-s', required=False, default=0.2, type=float, help='Test/Train split.')
    parser.add_argument('--dataset','-d', required=True, help='Dataset to to be downloaded and used. Currently supported are {mnist, fashion}.')
    parser.add_argument("--binarize", "-b", required=False, action='store_true', help="Trains a binarized neural network if true.")
    args = parser.parse_args()

    if args.dataset in ["mnist", "fashion"]:
        n_features, n_classes = 28*28,10
    else:
        print("Only {eeg, magic, mnist, fashion} is supported for the --dataset/-d argument but you supplied {}.".format(args.dataset))
        sys.exit(1)

   # model = SimpleCNN(n_features, n_classes, args.binarize, args.outpath)
    model = ResNet(BasicBlock, [2, 2, 2, 2], n_classes, args.outpath)

    implementations = [ 
        ("NHWC",{}) 
    ]

    if args.binarize:
        implementations.append( ("binary",{}) )

    optimizers = [
        ([None], [{}])
    ]

    performance = test_implementations(model = model, dataset= args.dataset, split = args.split, implementations = implementations, base_optimizers = optimizers, out_path = args.outpath, model_name = args.modelname)
    df = pd.DataFrame(performance)
    print(df)



if __name__ == '__main__':
    main()
