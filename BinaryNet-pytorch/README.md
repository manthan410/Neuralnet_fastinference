# BNN.pytorch
Binarized Neural Network (BNN) for pytorch
This is the pytorch version for the BNN code, fro VGG and resnet models
Please install torch and torchvision by following the instructions at: http://pytorch.org/
To run resnet18 for cifar10 dataset use: python main_binary.py --model resnet_binary --save resnet18_binary --dataset cifar10


The VGG models both Binary and non-Binary were trained for CIFAR10 and the onnx models were then generated to pass to the Fastinference Framework.



[1]https://github.com/itayhubara/BinaryNet.pytorch
