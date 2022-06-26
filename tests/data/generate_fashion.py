# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 18:54:36 2022

@author: manth
"""
from functools import cache
import sys
import os
import pandas as pd

import argparse

from sklearn.datasets import fetch_openml


def main():
    parser = argparse.ArgumentParser(
        description='Simple tool to generate datasets of varying difficulty / number of classes / number of features.')
    parser.add_argument('--out', required=True, help='Filename where data should written to.')
    parser.add_argument('--float', required=False, default=False, action='store_true',
                        help='True if features are float, else they are integer')

    args = parser.parse_args()

    print("Downloading Fashion MNIST")
    X, y = fetch_openml('Fashion-MNIST', return_X_y=True, as_frame=False, data_home=args.out, cache=True)
    # checking dtype

    print("dtype X is", X.dtype)
    # print("dtype y is",y.dtype)
    # X= X.astype('float16')
    # print(X.dtype)
    if not args.float:
        X = X.astype(int)
        # X = (X * 255).astype(int)

    XTrain, YTrain = X[:60000, :], y[:60000]
    # XTest, YTest = X[60000:,:], y[60000:]

    XTest, YTest = X[60000:, :], y[60000:]
    out_name = os.path.splitext(args.out)[0]

    print("Exporting data")
    dfTrain = pd.concat([pd.DataFrame(XTrain, columns=["f{}".format(i) for i in range(len(XTrain[0]))]),
                         pd.DataFrame(YTrain, columns=["label"])], axis=1)
    dfTest = pd.concat([pd.DataFrame(XTest, columns=["f{}".format(i) for i in range(len(XTrain[0]))]),
                        pd.DataFrame(YTest, columns=["label"])], axis=1)
    # dfTrain = pd.concat([pd.DataFrame(XTrain.astype('float16'), columns=["f{}".format(i) for i in range(len(XTrain[0]))]), pd.DataFrame(YTrain,columns=["label"])], axis=1)
    # dfTest = pd.concat([pd.DataFrame(XTest.astype('float16'), columns=["f{}".format(i) for i in range(len(XTrain[0]))]), pd.DataFrame(YTest,columns=["label"])], axis=1)
    # print(dfTrain)
    dfTrain.to_csv(os.path.join(out_name, "training.csv"), header=True, index=False)
    dfTest.to_csv(os.path.join(out_name, "testing.csv"), header=True, index=False)


if __name__ == '__main__':
    main()
