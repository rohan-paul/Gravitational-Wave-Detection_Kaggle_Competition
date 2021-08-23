In this competition you are provided with a training set of time series data containing simulated gravitational wave measurements from a network of 3 gravitational wave interferometers (LIGO Hanford, LIGO Livingston, and Virgo). Each time series contains either detector noise or detector noise plus a simulated gravitational wave signal. The task is to identify when a signal is present in the data (target=1).

So we need to use the training data along with the target value to build our model and make predictions on the test IDs in form of probability that the target exists for that ID.

So basically data-science helping here by building models to filter out this noises from data-streams (which includes both noise frequencies and Gravitational Waves frequencies) so we can single out frequencies for Gravitational-Waves.

---

# GO TO DATA TAB IN kAGGLE

Basic Description of the Data Provided

We are provided with a train and test set of time series data containing simulated gravitational wave measurements from a network of 3 gravitational wave interferometers:

LIGO Hanford

LIGO Livingston

Virgo

Each time series contains either detector noise or detector noise plus a simulated gravitational wave signal.

The task is to identify when a signal is present in the data (target=1).

Each .npy data file contains 3 time series (1 coming for each detector) and each spans 2 sec and is sampled at 2,048 Hz.

And we have a total of 5,60,000 files, each file of dimension of 3 * 4096, this turns out to be a huge time series


---


# Note on Keras Sequential model

There are two ways to build Keras models: sequential and functional.

The sequential API allows you to create models layer-by-layer for most problems. It is limited in that it does not allow you to create models that share layers or have multiple inputs or outputs. In short, you create a sequential model where you can easily add layers, and each layer can have convolution, max pooling, activation, drop-­out, and batch normalization.

Alternatively, the functional API allows you to create models that have a lot more flexibility as you can easily define models where layers connect to more than just the previous and next layers. In fact, you can connect layers to (literally) any other layer. As a result, creating complex networks such as siamese networks and residual networks become possible.


Here for this baseline Kernel I will be doing a Sequential Modle.

And acording to Keras documentation the Sequential model is a linear stack of layers. You can create a Sequential model by passing a list of layer instances to the constructor.


---

# Why I would need a Custom Data Generator Function for Keras Sequential Model building

The key reason is to be able to handle large data with batching, so the RAM/CPU/GPU does not need to handle the full data at once, which will anyway not be possible for this 72GB dataset.

So basically, since our code will in most cases be multicore-friendly, so we focus on doing more complex operations (e.g. computations from source files) without worrying about data generation becoming a bottleneck in the training process.

## Difference between fit() and fit_generator() in Keras

In Keras, using fit() and predict() is fine for smaller datasets which can be loaded into memory. But in practice, for most practical-use cases, almost all datasets are large and cannot be loaded into memory at once. The solution is to use fit_generator() and predict_generator() with custom data generator functions which can load data to memory during training or predicting.

In keras, fit() is much similar to sklearn's fit method, where you pass array of features as x values and target as y values. You pass your whole dataset at once in fit method. Also, use it if you can load whole data into your memory (small dataset).

In fit_generator(), you don't pass the x and y directly, instead they come from a generator. As it is written in keras documentation, generator is used when you want to avoid duplicate data when using multiprocessing. This is for practical purpose, when you have large dataset.