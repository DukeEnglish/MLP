#!/disk/scratch/mlp/miniconda2/bin/python
# coding: utf-8

# In[1]:


#import package
import os
import datetime
import numpy as np
import tensorflow as tf
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
import matplotlib.pyplot as plt
import time


# In[2]:

def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu, dropout = True, keep_prob=1.0):
    with tf.device('/cpu:0'):
        weights = tf.Variable(
            tf.truncated_normal(
                [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
            'weights')
        biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = tf.nn.dropout(nonlinearity(tf.matmul(inputs, weights) + biases), keep_prob)
    return outputs

def _variable_with_weight_decay(name, shape, stddev, wd):
    var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32),dtype=tf.float32)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


# In[3]:

train_data = CIFAR10DataProvider('train', batch_size=50)
valid_data = CIFAR10DataProvider('valid', batch_size=50)
inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
keep_prob=tf.placeholder(tf.float32)

kernel1_size=5
kernel2_size=5
kernel1_out=512
kernel2_out=512

with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights1',
                                         shape=[kernel1_size, kernel1_size, 3, kernel1_out],
                                         stddev=5e-2,
                                         wd=0.0)
    reshape = tf.reshape(inputs, [50,32,32,3])
    conv = tf.nn.conv2d(reshape, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases1', [kernel1_out], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)

# pool1
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
# norm1
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

# conv2
with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights2',
                                         shape=[kernel2_size, kernel2_size, kernel1_out, kernel2_out],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = biases = tf.get_variable('biases2', [kernel2_out], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(conv2)

# norm2
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
# pool2
pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

# local3
with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [50,-1])
    hidden_1 = fully_connected_layer(reshape, 32768, 32*32, keep_prob=0.8)

# local4
with tf.variable_scope('local4') as scope:
    outputs = fully_connected_layer(hidden_1, 32*32, train_data.num_classes, tf.identity, keep_prob=1.0)
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    train_step = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9).minimize(error)
    


# In[4]:

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    run_sum_time = time.time()
    for e in range(150):
        run_start_time = time.time()
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch, keep_prob:1.0})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
        print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                .format(valid_error, valid_accuracy))
        run_time = time.time() - run_start_time
        print ("time for each epoch",run_time)
    time_epochs=time.time()-run_sum_time
    print("sum time for epochs", time_epochs)


# In[ ]:



