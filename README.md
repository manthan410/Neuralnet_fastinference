# Neural-Net_fastinference


[1] https://github.com/sbuschjaeger/fastinference.git 

The fastinference framework [1] for Neural Networks were explored in this project for the in-depth understanding of its working and learning its usage for other deep and large model like VGG Net, Resnet, Alexnet. 


The Fastinference framework can generate the optimised C++ code for the target hardware (in our case CPU).
The primary goal was to observe the the inference accuracy and time after running the test cases. Generally, for resource constrained hardwares running the models which requires high storage, consume high enrergy and require heavy operation for double, float or int weight datatypes resulting from the models is a demerit. 
Thus, the Binarized Neural Network models are used that perform binary operations like XNOR and popcount on respective layers of the generated C++ code. The results obtained in the form of inference accuracy and time is compared  with corresponding double/ floating point networks.
The Binarized model ideally helps in achieving faster inference time and also saves energy with minor drop in accuracy resulting in being suitable to run the model on resource constrained hardwares.

For the project, fully connected and Convoluted network test cases used in [1] were explored. 

For eg., 
After following all the steps and installing all dependencies provided in the repository in [1]. The below line can be used to run the test case for fully-connected network.

```
$ python3 tests/test_mlp.py --outpath mnist-mlp  --dataset mnist
```
for corresponding binary model:
```
$ python3 tests/test_mlp.py --outpath mnist-bmlp  --dataset mnist --binarize
```
### BENCHMARK RESULTS
The benchmark results for 5 repitions for different models after the compilation of the generated C++ code is shown by the testCode.exe (application file) is as follows:

<img width="1061" alt="image" src="https://user-images.githubusercontent.com/94113767/165866805-80711b0f-6d47-4463-a2a5-b23441e24e88.png">

<img width="907" alt="image" src="https://user-images.githubusercontent.com/94113767/165867010-a8731379-ac68-457b-8793-828ae333c4ba.png">

### OBSERVATIONS
The results observed from the above benchmarks for the models trained on MNIST dataset shows that the Binarized fullyconnected (mlp) models are ~10 times slower than the floating model mainly of because of the number of trainable parameters /(weights+bias) i.e 136K and 6.6K parameters respectively. Also a accuracy difference of 2.28% between the the two models is observed. 
For the CNN, the Binarized models is ~2 times faster than the Floating network. The model size is 59.8KB compared to 0.3 MB of the Floating Network. Thus, BNNs being suitable to run on a resource contrained hardware.
Similar results can be observed for the models trained on Fashion MNIST with the exception of the accuracy being too low (in the range of 70s) for the shipped Binarized Neural Networks (BNNs) with the Framework.

Thus, there is need to explore better BNNs.
The onnx models of large and deep models like Alexnet, VGG16, Resnet18 were used from https://github.com/manthan410/models and passed to the Fastinference framework and was checked whether the framework can handle such large and deep networks. When passed through the framework the C++ files were generated and compiled successfully.
Further, VGGs for dataset like CIFAR10 was considered  then compared again with non-binary and Binary models. The results are observed by running different test cases provided in Vggf folder and BinaryNet-pytorch folder.




