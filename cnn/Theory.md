# Introduction to CNNs
In 2012 AlexNet architecture was proposed by Krizhevsky. 
- [View Paper](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- [View Code](https://github.com/akrizhevsky/cuda-convnet2/tree/master)

## Introduction
"To learn about thousands of objects from millions of images, we need a _model with a large learning capacity_. <br>
CNNs constitute one such class of models: _their capacity can be controlled by varying their depth and breadth_, 
and they also make strong and mostly correct assumptions about the nature of images 
(namely, stationarity of statistics and locality of pixel dependencies). <br>
Thus, compared to standard feedforward neural networks with similarly-sized layers, _CNNs have much fewer 
connections and parameters and so they are easier to train, while their theoretically-best performance is 
likely to be only slightly worse_.

## DataSet
ImageNet consists of variable-resolution images, while our system requires a constant input dimensionality. <br>
Therefore, we down-sampled the images to a fixed resolution of 256 × 256:
- we first rescaled the image such that the shorter side was of length 256 then we cropped out the central 256×256 patch from the resulting image. 
- we did not pre-process the images in any other way, except for subtracting the mean activity over the training set from each pixel. 
So we trained our network on the (centered) raw RGB values of the pixels.

## Architecture
AlexNet contains eight learned layers — 5 convolutional and 3 fully-connected.
![image](https://github.com/scifani/image-processing/assets/4973777/7e30dea8-2f82-4d1e-8b27-12b6951a22aa)

### ReLU Nonlinearity
Deep convolutional neural networks with ReLUs train several times faster than their equivalents with tanh units.
ReLUs have the desirable property that they do not require input normalization to prevent them from saturating. 
If at least some training examples produce a positive input to a ReLU, learning will happen in that neuron. 
However, we still find that the following local normalization scheme aids generalization.

### Overlapping Pooling
Pooling layers in CNNs summarize the outputs of neighboring groups of neurons in the same kernel map.

### Overall Architecture
The net contains 8 layers with weights: the first 5 are convolutional and the remaining 3 are fullyconnected. <br>
The output of the last fully-connected layer is fed to a 1000-way softmax which produces a distribution over the 1000 class labels. <br>
Our network maximizes the multinomial logistic regression objective, which is equivalent to maximizing the 
average across training cases of the log-probability of the correct label under the prediction distribution. <br>
The ReLU non-linearity is applied to the output of every convolutional and fully-connected layer.
