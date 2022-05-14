from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):   # remember change name
    """
    Performs 2D convolution given 4D inputs and filter Tensors.
    :param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
    :param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
    :param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
    :param padding: either "SAME" or "VALID", capitalization matters
    :return: outputs, Tensor with shape [num_examples, output_height, output_width, output_channels]
    """
    ## Alert
    assert inputs.shape[3] == filters.shape[2], "The number of input in channels are not the same as the filter's in channels"
    
    ## Inputs
    num_examples = inputs.shape[0]
    in_height = inputs.shape[1]
    in_width = inputs.shape[2]
    input_in_channels = inputs.shape[3]
    
    ## Filters
    filter_height = filters.shape[0]
    filter_width = filters.shape[1]
    filter_in_channels = filters.shape[2]
    filter_out_channels = filters.shape[3]
    
    ## Strides
    num_examples_stride = strides[0]
    strideY = strides[1]
    strideX = strides[2]
    channels_stride = strides[3]
    
    ## Padding
    if padding == "SAME":
        pad_size = math.floor((filter_height-1)/2)
    else:
        pad_size = 0
    
    
    ## calculating the output shape
    w_c = int((in_width - filter_width + 2*pad_size)/strideX + 1)
    h_c = int((in_height- filter_height+ 2*pad_size)/strideY + 1)
    d_c = int(filter_out_channels)
    n_c = int(num_examples)
    
    # creat output structure
    output_layer = np.zeros([n_c,h_c,w_c,d_c])
    
    # create stride size
    # convolution
    for i in range(n_c):                                                             # Every sample                                      
        for j in range(input_in_channels):                                           # Every chanel
            pad_layer = np.pad(inputs[i,:,:,j],pad_size)                         
            filter_layer = filters[:,:,j,:]                                        
            for k in range(d_c):                                                     # Every out chanel      #checked!
                for ii in range(h_c):                                                # Vertical convolution
                    for jj in range(w_c):                                            # Horizontal convolution
                        padd = pad_layer[ii:ii+filter_height,jj:jj+filter_width]
                        filt = filter_layer[:,:,k]
                        output_layer[i,ii,jj,k] += np.sum(padd*filt)
    return tf.convert_to_tensor(output_layer, dtype = tf.float32) ## tf.float32

def same_test_0():
	'''
	Simple test using SAME padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="SAME")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="SAME")
	print("SAME_TEST_0:", "my conv2d:", my_conv[0][0][0], "tf conv2d:", tf_conv[0][0][0].numpy())

def valid_test_0():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.double)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_0:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_1():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[3,5,3,3],[5,1,4,5],[2,5,0,1],[3,3,2,1]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = tf.Variable(tf.random.truncated_normal([3, 3, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_2():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[1,3,2,1],[1,3,3,1],[2,1,1,3],[3,2,3,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = np.array([[1,2,3],[0,1,0],[2,1,2]]).reshape((3,3,1,1)).astype(np.float32)
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def main():
    # TODO: Add in any tests you may want to use to view the differences between your and TensorFlow's output
    same_test_0()
    valid_test_0()
    valid_test_1()
    valid_test_2()

if __name__ == '__main__':
	main()
