#!/disk/scratch/mlp/miniconda2/bin/python
# coding: utf-8

# ## Part1: Depths

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import tarfile

from six.moves import urllib

import os
import tensorflow as tf
import numpy as np
from mlp.data_providers import CIFAR10DataProvider
import matplotlib.pyplot as plt
import time


# In[2]:

random_seed = 187556
rng = np.random.RandomState(random_seed)

train_data = CIFAR10DataProvider('train', batch_size=50, rng=rng)
valid_data = CIFAR10DataProvider('valid', batch_size=50, rng=rng)


# In[3]:

def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs


# ### Two layers

# In[4]:

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      #tf.truncated_normal_initializer(stddev=stddev))
      tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


# In[5]:

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


# In[6]:

train_data.reset()
valid_data.reset()
inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 200


with tf.variable_scope('conv1') as scope:
    inputs_new=tf.reshape(inputs,[50,32,32,3])
    kernel1 = _variable_with_weight_decay('conv_weights_1', shape=[3, 3, 3, 64], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv2d(inputs_new, kernel1, [1, 1, 1, 1], padding='SAME')
    biases1 = _variable_on_cpu('biases_1', [64], tf.constant_initializer(0.0))
    pre_activation1 = tf.nn.bias_add(conv, biases1)
    conv1 = tf.nn.relu(pre_activation1, name=scope.name)
    
with tf.variable_scope('conv2') as scope:
    kernel2 = _variable_with_weight_decay('conv_weights_2', shape=[3, 3, 64, 64], stddev=5e-2, wd=0.0)
    conv_2 = tf.nn.conv2d(conv1, kernel2, [1, 1, 1, 1], padding='SAME')
    biases2 = _variable_on_cpu('biases_2', [64], tf.constant_initializer(0.0))
    pre_activation2 = tf.nn.bias_add(conv_2, biases2)
    conv2 = tf.nn.relu(pre_activation2, name=scope.name)


    

pool1 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')


#norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')


# In[7]:

num_hidden = 1000
conv2_out=tf.reshape(pool1,[50, -1])
with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer(conv2_out, 16384, num_hidden)
with tf.name_scope('output-layer'):
    outputs = fully_connected_layer(hidden_1, num_hidden, train_data.num_classes, tf.identity)

with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(error)
    
init = tf.global_variables_initializer()


# In[8]:

train_data.reset()
valid_data.reset()
running_error_one = []
running_accuracy_one = []
valid_error_one = []
valid_accuracy_one = []

with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        start_time = time.time()
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        epoch_time = time.time() - start_time
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        running_error_one.append(running_error)
        running_accuracy_one.append(running_accuracy)
        
        print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f} epoch time={3:.2f}'
              .format(e + 1, running_error, running_accuracy, epoch_time))
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        valid_error_one.append(valid_error)
        valid_accuracy_one.append(valid_accuracy)
            
        print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                .format(valid_error, valid_accuracy))


# In[ ]:

fig_1 = plt.figure(figsize=(12, 6))
ax1 = fig_1.add_subplot(111)
ax1.plot(running_accuracy_one, label = 'train_acc_one')
ax1.plot(valid_accuracy_one, label = 'valid_acc_one')
#ax1.plot(running_accuracy_two, label = 'train_acc_two')
#ax1.plot(running_accuracy_three, label = 'train_acc_three')
#ax1.plot(running_accuracy_four, label = 'train_acc_four')
#ax1.plot(running_accuracy_five, label = 'train_acc_five')
#ax1.plot(running_accuracy_six, label = 'train_acc_six')
ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training set accuracy')

fig_2 = plt.figure(figsize=(12, 6))
ax2 = fig_2.add_subplot(111)
ax2.plot(running_error_one, label = 'train_err_one')
ax2.plot(valid_error_one, label = 'valid_acc_one')
#ax2.plot(running_error_two, label = 'train_err_two')
#ax2.plot(running_error_three, label = 'train_err_three')
#ax2.plot(running_error_four, label = 'train_err_four')
#ax2.plot(running_error_five, label = 'train_err_five')
#ax2.plot(running_error_six, label = 'train_err_six')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Training set error')


plt.show()


# In[ ]:




# In[ ]:



