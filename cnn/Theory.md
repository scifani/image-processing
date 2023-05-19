## Introduction
ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. 
These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.

To learn about thousands of objects from millions of images, we need a _model with a large learning capacity_. <br>
CNNs constitute one such class of models: _their capacity can be controlled by varying their depth and breadth_, 
and they also make strong and mostly correct assumptions about the nature of images 
(namely, stationarity of statistics and locality of pixel dependencies). <br>
Thus, compared to standard feedforward neural networks with similarly-sized layers, _CNNs have much fewer 
connections and parameters and so they are easier to train, while their theoretically-best performance is 
likely to be only slightly worse_.

In particular, unlike a regular Neural Network, the layers of a ConvNet have neurons arranged in 3 dimensions: width, height, depth.

![CNN](./asset/images/cnn.jpeg)

ConvNet is a sequence of layers, and every layer of a ConvNet transforms one volume of activations to another through a differentiable function. 
We use four main types of layers to build ConvNet architectures: 
- __Convolutional__ Layer: computes the output of neurons that are connected to local regions in the input, 
  each computing a dot product between their weights and a small region they are connected to in the input volume.
- __ReLU__ Layer: applies an elementwise activation function, such as the max(0,x) thresholding at zero. This leaves the size of the volume unchanged.
- __Pooling__ Layer: performs a downsampling operation along the spatial dimensions (width, height).
- __Fully-Connected__ Layer (exactly as seen in regular Neural Networks): the last layer computes the class scores, resulting in volume of size 1 × 1 × _C_, where _C_ is the numeber of classes.

### Local Connectivity
When dealing with high-dimensional inputs such as images, as we saw above it is impractical to connect neurons to all neurons in the previous volume. 
Instead, we will connect each neuron to only a local region of the input volume. 
The spatial extent of this connectivity is a hyperparameter called the __receptive field__ of the neuron (equivalently this is the __filter size__). 
The extent of the connectivity along the depth axis is always equal to the depth of the input volume. 
It is important to emphasize again this asymmetry in how we treat the spatial dimensions (width and height) and the depth dimension: 
the connections are _local in 2D space_ (along width and height), but always _full along the depth_ of the input volume.

The dimension of the filter affects the dimension of the output volume in the following way:
|  Input Volume   |     Weights     | Output Volume |
| --------------- | --------------- | ----------------------------------------------------------------------------------------- |
| $$n × n × n_c$$ | $$f × f × n_c$$ | $$\left( {n + 2p -f \over s} + 1 \right) × \left( {n + 2p -f \over s} + 1 \right) × n_c'$$ |

(with $`n_c'`$ as the number of filters, which are detecting different features)

Local Connectivity             |  Neuron Model
:-------------------------:|:-------------------------:
![depthcol](./asset/images/depthcol.jpeg)  |  ![neuron_model](./asset/images/neuron_model.jpeg)

[Stanford cs231n Notes](https://cs231n.github.io/convolutional-networks/)

## AlexNet
In 2012 AlexNet architecture was proposed by Krizhevsky. 
- [View Paper](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- [View Code](https://github.com/akrizhevsky/cuda-convnet2/tree/master)

### Architecture
![AlexNet](./asset/images/AlexNet.png)

The net contains 8 layers with weights: the first 5 are convolutional and the remaining 3 are fullyconnected.
- The first convolutional layer has 96 kernels of size 11×11×3 with a stride of 4 pixels. It takes as input the 224×224×3 image. <br>
- The second convolutional layer has 256 kernels of size 5×5×48. It takes as input the (response-normalized and pooled) output of the first convolutional layer. 
  Pooling layers in CNNs summarize the outputs of neighboring groups of neurons in the same kernel map.
- The third convolutional layer has 384 kernels of size 3×3×256 connected to the (normalized, pooled) outputs of the second convolutional layer.
- The fourth convolutional layer has 384 kernels of size 3×3×192
- The fifth convolutional layer has 256 kernels of size 3×3×192. 
- The final three fully-connected layers have 4096 neurons each.
- The output of the last fully-connected layer is fed to a 1000-way softmax which produces a distribution over the 1000 class labels.

The net maximizes the multinomial logistic regression objective, which is equivalent to maximizing the 
average across training cases of the log-probability of the correct label under the prediction distribution.

The ReLU non-linearity is applied to the output of every convolutional and fully-connected layer. 
ReLUs train several times faster than their equivalents with tanh units and have the desirable property that 
they do not require input normalization to prevent them from saturating. 
However, some local normalization is still applied since we find that the it aids generalization.

### DataSet
ImageNet consists of variable-resolution images, while our system requires a constant input dimensionality. <br>
Therefore, we down-sampled the images to a fixed resolution of 256 × 256:
- we first rescaled the image such that the shorter side was of length 256 then we cropped out the central 256×256 patch from the resulting image. 
- we did not pre-process the images in any other way, except for subtracting the mean activity over the training set from each pixel. 
So we trained our network on the (centered) raw RGB values of the pixels.

### Reducing Overfitting
#### Data Augmentation
The easiest and most common method to __reduce overfitting__ on image data is to artificially enlarge
the dataset using label-preserving transformations.
- The first form of data augmentation consists of generating image translations and horizontal reflections. 
  We do this by extracting random 224 × 224 patches (and their horizontal reflections) from the 256×256 
  images and training our network on these extracted patches. This is the reason why the input images in Figure are 224 × 224 × 3-dimensional.
- The second form of data augmentation consists of altering the intensities of the RGB channels in training images. 
  Specifically, we perform PCA on the set of RGB pixel values throughout the ImageNet training set. 
  This scheme approximately captures an important property of natural images, namely, that object identity is 
  invariant to changes in the intensity and color of the illumination.
  
At test time, the network makes a prediction by extracting five 224 × 224 patches (the four corner patches 
and the center patch) as well as their horizontal reflections (hence ten patches in all), and averaging the 
predictions made by the network’s softmax layer on the ten patches.

#### Dropout
Dropout consists of setting to zero the output of each hidden neuron with probability 0.5. The neurons which are
“dropped out” in this way do not contribute to the forward pass and do not participate in backpropagation. 
So every time an input is presented, the neural network samples a different architecture, but all these architectures share weights. 
This technique reduces complex co-adaptations of neurons, since a neuron cannot rely on the presence of particular other neurons. 
It is, therefore, forced to learn more robust features that are useful in conjunction with many different random subsets of the other neurons. 

At test time, we use all the neurons but multiply their outputs by 0.5, which is a reasonable approximation to taking 
the geometric mean of the predictive distributions produced by the exponentially-many dropout networks.

We use dropout in the first two fully-connected layers. Without dropout, our network exhibits substantial overfitting. 
However, Dropout roughly doubles the number of iterations required to converge.

### Training details
We trained our models using stochastic gradient descent with a batch size of 128 examples, momentum of 0.9, and weight decay of 0.0005. 
We found that this small amount of weight decay was important for the model to learn.
