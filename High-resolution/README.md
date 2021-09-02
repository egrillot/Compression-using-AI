# High Resolution

## Introduction

Our goal here is first to reduce the high-resolution image into a low-resolution image (basically we downscale a (2048, 2048, 3) array in a (512, 512, 3) array). Then we develop
an extension of the 32x32x3 compressor for images with shape (512, 512, 3). The low-resolution image will be compressed in 2 vectors of shape (32, 32, 128) and (256, 128). Let's 
summary what we are going to do : as input we get a 12 509 292 dimensional vector that we want to compress into a 163 840 dimensional vector. 

For the first step, we are going to use a RRDN structure :

<table>
  <tr>
    <td>High-resolution -> Low-resolution</td>
  </tr>
  <tr>
    <td><img src="images/HR-_LR.png"></td>
  </tr>
  <tr>
    <td>Low-resolution -> High-resolution</td>
  </tr>
  <tr>
    <td><img src="images/LR-_HR.png"></td>
  </tr>
</table>

and train both model on the div2K dataset (using only the LRx4 bicubic package). Remind for later that this stage is very demanding on RAM resources ! The second step is the same as what we did for 32x32x3 compression.

## Results

For the first step, a pre-trained model is available [here](https://github.com/idealo/image-super-resolution/tree/master/weights/sample_weights)

We reach a mse of 0.0044 in 400 epochs for the autoencoding of low-resolution images as you can see below :

<img src="images/training.png">

And finally we have succesfully reach our goal :

<table>
  <tr>
    <td>High-resolution</td>
    <td>Low-resolution</td>
    <td>Encoded image</td>
    <td>Decoded image</td>
    <td>Upscaled image</td>
  </tr>
  <tr>
    <td><img src="images/test_HR.png"></td>
    <td><img src="images/test_LR.png"></td>
    <td><img src="images/test_encoded_1.png"> <img src="images/test_encoded_2.png"></td>
    <td><img src="images/test_decoded.png"></td>
    <td><img src="images/test_HR_end.png"></td>
  </tr>
  <tr>
    <td><sub><sup>shape (1356, 2040, 3)</sup></sub></td>
    <td><sub><sup>shape (339, 510, 3)</sup></sub></td>
    <td><sub><sup>shape (32, 32, 128) & (256, 128, 3)</sup></sub></td>
    <td><sub><sup>shape (339, 510, 3)</sup></sub></td>
    <td><sub><sup>shape (1356, 2040, 3)</sup></sub></td>
  </tr>
</table>

## Usage

Let's start by using the ``div2k`` class available in the file **data.py**. After reading the LICENSE and downloading the div2K dataset, you can use the following command line :

~~~
dataset = div2k()
dataset.process(path/to/the/dataset)
~~~

Then you can get your data as follows :

~~~
High_resolution_train, Low_resolution_train, High_resolution_validation, Low_resolution_validation = dataset.get_data()
~~~

Also you can save and load your processed data with the method ``.save_data(directory/path)`` and ``.load_data(directory/path)``. Now you can train your RRDN model, lets instance it with the ``RRDN`` class available in the fle **model.py** :

~~~
rrdn = RRDN(rdb_blocks_number=6, conv_rdb_number=6, rdb_filter=64, input_shape=(512, 512, 3), task='HR->LR')
~~~

Here we have 6 [RDB](https://sh-tsang.medium.com/review-rdn-residual-dense-network-super-resolution-9738b8ce51e2) (Residual Dense Block) containing each 6 convolutional layers with 64 map features. This model will reduce a high-resolution image into a low-resolution image but you can do the opposite by changing the ``task`` parameter by ``'LR->HR'`` and adapte your ``input_shape`` parameter. 
