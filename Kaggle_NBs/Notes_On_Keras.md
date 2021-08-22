In Keras, all the nodes of a layer can be initialized by simply initializing the layer itself. The individual
operation of a generalized layer node can be seen in the following diagram. At each
node, the input data is multiplied by a set of weights using matrix multiplication. The sum of the product between the weights and the
input is applied, which may or may not include a bias, as shown by the input node equal
to 1 in the following diagram. Further functions may be applied to the output of this
matrix multiplication, such as activation functions:

![](assets/2021-08-21-21-38-30.png)


## Note on Keras Sequential model

There are two ways to build Keras models: sequential and functional.

**The sequential API** allows you to create models layer-by-layer for most problems. It is limited in that it does not allow you to create models that share layers or have multiple inputs or outputs.
In short, you create a sequential model where you can easily add layers, and each layer can have convolution, max pooling, activation, drop-­out, and batch normalization.

Alternatively, **the functional API** allows you to create models that have a lot more flexibility as you can easily define models where layers connect to more than just the previous and next layers. In fact, you can connect layers to (literally) any other layer. As a result, creating complex networks such as siamese networks and residual networks become possible.

From the definition of Keras documentation the Sequential model is a linear stack of layers. You can create a Sequential model by passing a list of layer instances to the constructor. The common architecture of ConvNets is a sequential architecture. However, some architectures are not linear stacks. For example, siamese networks are two parallel neural networks with some shared layers.

With the Sequential Models, you need to ensure the
input layer has the right number of inputs. Assume that you have 3,072
input variables; then you need to create the first hidden layer with 512
nodes/neurons. In the second hidden layer, you have 120 nodes/neurons.
Finally, you have ten nodes in the output layer. For example, an image
maps onto ten nodes that shows the probability of being label1 (airplane),
label2 (automobile), label3 (cat), ..., label10 (truck). The node of highest
probability is the predicted class/label.


### Some common layer types in Keras are as follows:

**Dense**: This is a fully connected layer in which all the nodes of the layer are
directly connected to all the inputs and all the outputs. ANNs for classification or
regression tasks on tabular data usually have a large percentage of their layers with
this type in the architecture.

**Convolutional**: This layer type creates a convolutional kernel that is convolved with
the input layer to produce a tensor of outputs. This convolution can occur in one
or multiple dimensions. ANNs for the classification of images usually feature one or
more convolutional layers in their architecture.

**Pooling**: This type of layer is used to reduce the dimensionality of an input layer.
Common types of pooling include max pooling, in which the maximum value of
a given window is passed through to the output, or average pooling, in which
the average value of a window is passed through. These layers are often used
in conjunction with a convolutional layer, and their purpose is to reduce the
dimensions of the subsequent layers, allowing for fewer training parameters to be
learned with little information loss.

**Recurrent**: Recurrent layers learn patterns from sequences, so each output is
dependent on the results from the previous step. ANNs that model sequential data
such as natural language or time-series data often feature one or more recurrent
layer types.


There are other layer types in Keras; however, these are the most common types when
it comes to building models using Keras.


---

## Why I would need a Custom Data Generator Function for Keras Sequential Model building

The key reason is to be able to handle large data with batching, so the RAM/CPU/GPU does not need to handle the full data at once, which will anyway not be possible for this 72GB dataset.

So basically, since our code will in most cases be multicore-friendly, so we focus on doing more complex operations (e.g. computations from source files) without worrying about data generation becoming a bottleneck in the training process.


**DataGenerator(Sequence)** => Now, let's go through the details of how to set the Python class DataGenerator, which will be used for real-time data feeding to your Keras model. We make `DataGenerator` inherit the properties of `keras.utils.Sequence` so that we can leverage nice functionalities such as multiprocessing.

While we have built-in Data Generator like [ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator), we still need a plethora of custom Generator function. Because, Model training is not limited to a single type of input and target. There are times when a model is fed with multiple types of inputs at once. For example, say in a multi-modal classification problem which needs to process text and image data simultaneously. Here, obviously we cannot use ImageDataGenerator. Hence, we need a custom data generator.

According to [Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence) - Every Sequence must implement the `__getitem__` and the `__len__` methods. If you want to modify your dataset between epochs you may implement on_epoch_end. The method __getitem__ should return a complete batch.

 ### A note on `yield` function with respect to the custom data generator here for Sequence API of Keras

Here I am using the Sequence API, which works a bit different than plain generators. In a generator function, you would use the `yield` keyword to perform iteration inside a while True: loop, so each time Keras calls the generator, it gets a batch of data and it automatically wraps around the end of the data.

But in a Sequence-API, there is an index parameter to the `__getitem__` function, so no iteration or `yield` is required, this is performed by Keras for you. This is made so the sequence can run in parallel using multiprocessing, which is not possible with old generator functions.

---

### Difference between fit() and fit_generator() in Keras

In Keras, using fit() and predict() is fine for smaller datasets which can be loaded into memory. But in practice, for most practical-use cases, almost all datasets are large and cannot be loaded into memory at once. The solution is to use fit_generator() and predict_generator() with custom data generator functions which can load data to memory during training or predicting.

In keras, fit() is much similar to sklearn's fit method, where you pass array of features as x values and target as y values. You pass your whole dataset at once in fit method. Also, use it if you can load whole data into your memory (small dataset).

In fit_generator(), you don't pass the x and y directly, instead they come from a generator. As it is written in keras documentation, generator is used when you want to avoid duplicate data when using multiprocessing. This is for practical purpose, when you have large dataset.

