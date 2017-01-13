#!/usr/bin/env python
__author__="smafjal"

import tensorflow as tf

# parameter
DISP_CONSOLE=True
ALPHA=0.1

def init_weights(idx,_shape):
    return tf.Variable(tf.truncated_normal(_shape, stddev=0.01),name=str(idx)+"_weights")

def init_bias(idx,_shape):
    return tf.Variable(tf.constant(0.1, shape=_shape),name=str(idx)+"_bias")

def pool_layer(idx,_X):
    return tf.nn.max_pool(_X, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def drop_layer(idx,_X,p_keep_conv):
    return tf.nn.dropout(_X, p_keep_conv,name=str(idx)+"_dropout")

def fc_layer(idx,_inputs,hiddens,flat=False,linear=False):
    name="fully_connected_"+str(idx)
    with tf.name_scope(name):
        input_shape = _inputs.get_shape().as_list()
        if flat:
            dim = input_shape[1]*input_shape[2]*input_shape[3]
            inputs_transposed = tf.transpose(_inputs,(0,3,1,2))
            inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
        else:
            dim = input_shape[1]
            inputs_processed = _inputs

        _W=init_weights(idx,[dim,hiddens])
        _B=init_bias(idx,[hiddens])

        if DISP_CONSOLE:
            print 'Layer %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' %(idx,hiddens,int(dim),int(flat),1-int(linear))

        if linear:
            return tf.add(tf.matmul(inputs_processed,_W),_B,name=str(idx)+'_fc')
        logits = tf.add(tf.matmul(inputs_processed,_W),_B)
        return tf.maximum(ALPHA*logits,logits,name=str(idx)+'_logits')

def conv_layer(idx,_X,filters,filter_size,stride):
    name="convolution_"+str(idx)
    with tf.name_scope(name):
        channels=_X.get_shape()[3]
        _W=init_weights(idx,[filter_size,filter_size,int(channels),filters])
        _B=init_bias(idx,[filters])
        if DISP_CONSOLE:
            print 'Layer %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx,filter_size,filter_size,stride,filters,int(channels))

        conv=tf.nn.conv2d(_X, _W,strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_conv_op')
        conv_bias=tf.nn.bias_add(conv,_B,name=str(idx)+'_bias_add')
        return tf.nn.relu(conv_bias,name=str(idx)+"_relu")

def build_model(X,Y,hidden_shape,keep_prob):
    print "Input-X-Shape: ",X.get_shape()
    print "Input-Y-Shape: ",Y.get_shape()

    num_class=Y.get_shape().as_list()[-1]
    conv_1=conv_layer(1,X,32,3,1)
    pool_2=pool_layer(2,conv_1)
    drop_3=drop_layer(3,pool_2, keep_prob)

    conv_4=conv_layer(4,drop_3,64,3,1)
    pool_5=pool_layer(5,conv_4)
    drop_6=drop_layer(6,pool_5,keep_prob)

    conv_7=conv_layer(7,drop_6,128,3,1)
    pool_8=pool_layer(8,conv_7)
    final_9=fc_layer(9,pool_8,hidden_shape,flat=True,linear=False)
    final_10=fc_layer(10,final_9,num_class,flat=False,linear=True)
    return final_10

def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    loss = tf.reduce_mean(cross_entropy)
    return loss

def training(loss, learning_rate=5e-3):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    predictions = tf.argmax(logits, 1, name='predictions')
    correct_predictions = tf.equal(predictions,tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions,'float'), name='accuracy')
    return accuracy

def input_xy(x_shape,y_shape):
    X=tf.placeholder(tf.float32,x_shape)
    Y=tf.placeholder(tf.float32,y_shape)
    return X,Y
