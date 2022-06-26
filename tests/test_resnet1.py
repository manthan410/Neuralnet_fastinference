import torch
#from torch import nn
#from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import onnx

import torch.nn as nn
import torch.nn.functional as F

#import pytorch_lightning as pl

import json
import pandas as pd
import os
import argparse
from sklearn.metrics import accuracy_score

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

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # FROM 3 TO 1
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

    def predict(self, X):
        return self.forward(torch.from_numpy(X).float()).argmax(axis=1)


def train_store(model, x_train, y_train, x_test, y_test, out_path, name, loss_fn, optimiser, device, epochs):
    '''training'''

    x_train_t = torch.from_numpy(x_train).float()
    print(x_train_t.dtype)
    y_train_t = torch.from_numpy(y_train).long()
    print(y_train_t.dtype)

    train_dataloader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=64)
    # val_loader = None
    # trainer = torch.Trainer(max_epochs = 10, default_root_dir = out_path, progress_bar_refresh_rate = 0)
    # trainer.fit(model, train_dataloader, val_loader)
    # model.eval()
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        for input, target in train_dataloader:
            input, target = input.to(device), target.to(device)

            # calculate loss
            prediction = model(input)
            loss = loss_fn(prediction, target)

            # backpropagate error and update weights
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        print(f"loss: {loss.item()}")
        print("---------------------------")
    print("Finished training")

    '''storing'''
    dummy_x = torch.randn(1, x_train.shape[1], requires_grad=False)
    # print(dummy_x.dtype)
    preds = model.predict(x_test)
    accuracy = accuracy_score(y_test, preds) * 100.0
    djson = {
        "accuracy": accuracy,
        "name": name
    }
    print("accuracy: {}".format(djson["accuracy"]))
    # print("batch-latency: {}".format(djson["batch-latency"]))
    # print("single-latency: {}".format(djson["single-latency"]))

    with open(os.path.join(out_path, name + ".json"), "w") as outfile:
        json.dump(djson, outfile)

    onnx_path = os.path.join(out_path, name + ".onnx")
    print("Exporting {} to {}".format(name, onnx_path))

    # Export the model
    torch.onnx.export(model, dummy_x, onnx_path, export_params=True, opset_version=11, do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    onnx_model = onnx.load(os.path.join(out_path, name + ".onnx"))
    checker = onnx.checker.check_model(onnx_model)
    print(checker)
    return onnx_path

def main():
    parser = argparse.ArgumentParser(
        description='Train MLPs on the supplied data. This script assumes that each supplied training / testing CSV has a unique column called `label` which contains the labels.')
    parser.add_argument('--training', required=True, help='Filename of training data CSV-file')
    parser.add_argument('--testing', required=True, help='Filename of testing data CSV-file')
    parser.add_argument('--out', required=True, help='Folder where data should be written to.')
    parser.add_argument('--name', required=True, help='Modelname')

    args = parser.parse_args()

    # use generated dataset like MNIST

    print("Loading training data")
    df = pd.read_csv(args.training)
    y_train = df["label"].to_numpy().astype('int8')
    x_train = df.drop(columns=["label"]).to_numpy().astype('float16')
    print("X-train", x_train.dtype)
    print("Y-train", y_train.dtype)

    print("Loading testing data")
    df = pd.read_csv(args.testing)
    y_test = df["label"].to_numpy()
    x_test = df.drop(columns=["label"]).to_numpy()
    print("")

    n_classes = len(set(y_train) | set(y_test))
    # print(len(y_train))
    # print(len(y_test))
    # print(n_classes)
    #input_dim = x_train.shape[1]
    # print(input_dim)
    # print(y_train)

    # construct model and assign it to device
    # if torch.cuda.is_available():
    # device = "cuda"
    # else:
    device = "cpu"
    print(f"Using {device}")

    # Intialise network

    #model = NN2(input_dim, n_classes)
    model = ResNet(BasicBlock, [2, 2, 2, 2], n_classes)
    print(model)

    # initialise loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=0.001)

    # train model and store model
    epochs = 1
    train_store(model, x_train, y_train, x_test, y_test, args.out, args.name, loss_fn, optimiser, device, epochs)
    print("")


if __name__ == '__main__':
    main()