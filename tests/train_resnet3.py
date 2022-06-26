
import torch
from torchvision.models import resnet18
from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST

from torchvision.transforms import ToTensor

import pytorch_lightning as pl
#from pytorch_lightning.core.decorators import auto_move_data
import json
import pandas as pd
import os
import argparse

class ResNetMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=10)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_no):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.005)


def main():
    parser = argparse.ArgumentParser(
        description='Train MLPs on the supplied data. This script assumes that each supplied training / testing CSV has a unique column called `label` which contains the labels.')
   # parser.add_argument('--training', required=True, help='Filename of training data CSV-file')
   # parser.add_argument('--testing', required=True, help='Filename of testing data CSV-file')
    parser.add_argument('--out', required=True, help='Folder where data should be written to.')
    parser.add_argument('--name', required=True, help='Modelname')

    args = parser.parse_args()

    train_ds = MNIST("mnist", train=True, download=True, transform=ToTensor())
   # train_ds =
    test_ds = MNIST("mnist", train=False, download=True, transform=ToTensor())
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64)

    model = ResNetMNIST()
    trainer = pl.Trainer(
        max_epochs=1,
        progress_bar_refresh_rate=20
    )

    trainer.fit(model, train_dl)
    print("Finished training")

    '''storing'''
    dummy_x = torch.randn(1,784, requires_grad=False)
    # print(dummy_x.dtype)
   # preds = model.predict(x_test)
  #  accuracy = accuracy_score(y_test, preds) * 100.0
    #djson = {
   #     "accuracy": accuracy,
    #    "name": name
    #}
  #  print("accuracy: {}".format(djson["accuracy"]))
    # print("batch-latency: {}".format(djson["batch-latency"]))
    # print("single-latency: {}".format(djson["single-latency"]))

   # with open(os.path.join(args.out, args.name + ".json"), "w") as outfile:
    #    json.dump(djson, outfile)

    onnx_path = os.path.join(args.out, args.name + ".onnx")
    print("Exporting {} to {}".format(args.name, onnx_path))

    # Export the model
    torch.onnx.export(model, dummy_x, onnx_path, export_params=True, opset_version=11, do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    onnx_model = onnx.load(os.path.join(out, args.name + ".onnx"))
    checker = onnx.checker.check_model(onnx_model)
    print(checker)


if __name__ == '__main__':
    main()