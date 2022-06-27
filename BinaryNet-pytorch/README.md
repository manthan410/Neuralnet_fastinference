# BNN.pytorch
Binarized Neural Network (BNN) for pytorch
[1]This is the pytorch version for the BNN code, for VGG and resnet and Alexnet models.
Please install torch and torchvision by following the instructions at: http://pytorch.org/
To run resnet18 for cifar10 dataset use: python main_binary.py --model resnet_binary --save resnet18_binary --dataset cifar10


The VGG models both Binary and non-Binary were trained for CIFAR10. The generated onnx models are then used to pass through the Fastinference Framework.

The generated ONNX files can be visualised using https://netron.app/

[1]https://github.com/itayhubara/BinaryNet.pytorch
