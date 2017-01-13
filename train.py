import tensorflow as tf
import cnn_digit_cls as model
import numpy as np
import sys
import os
import time
import data_reader as reader
from tensorflow.examples.tutorials.mnist import input_data

# Model Parameter
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUM_CHANNELS = 1
NUM_CLASS=10
HIDDEN_SHAPE=625
BATCH_SIZE = 50
NUM_EPOCHS = 10
DISP_STEP=1

# Data dir
data_dir="data_dir"
saved_model_dir="saved_model"
logs_dir="logs"

def train(data_dir,saved_model_dir,logs_dir):
    input_x_shape=[None,IMAGE_WIDTH,IMAGE_HEIGHT,NUM_CHANNELS]
    input_y_shape=[None,NUM_CLASS]

    X,Y=model.input_xy(input_x_shape,input_y_shape)
    keep_prob = tf.placeholder("float")

    logits = model.build_model(X,Y,HIDDEN_SHAPE,keep_prob)
    loss = model.loss(logits, Y)
    train_op = model.training(loss, learning_rate=5e-2)
    accuracy = model.evaluation(logits, Y)

    global_init = tf.global_variables_initializer()
    locla_init=tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(global_init)
        sess.run(locla_init)
        saver = tf.train.Saver(tf.global_variables())
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        for epoch in range(1,NUM_EPOCHS):
            avg_cost=0
            total_batch = int(mnist.train.num_examples/BATCH_SIZE)
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
                batch_x = batch_x.reshape(-1,IMAGE_WIDTH,IMAGE_HEIGHT,NUM_CHANNELS)
                batch_y = batch_y.reshape(-1,NUM_CLASS)

                feed_dict={X: batch_x,Y: batch_y, keep_prob: 0.8}
                sess.run(train_op, feed_dict)

            if epoch % DISP_STEP == 0:
                batch_x, batch_y = mnist.test.images,mnist.test.labels
                batch_x = batch_x.reshape(-1,IMAGE_WIDTH,IMAGE_HEIGHT,NUM_CHANNELS)
                batch_y = batch_y.reshape(-1,NUM_CLASS)

                feed_dict={X: batch_x,Y:batch_y,keep_prob:0.8}
                accuracy_val=sess.run(accuracy,feed_dict)
                print "Step: %d Accuracy: %f" % (epoch,accuracy_val)

        print "Learning Complete!"

        batch_x, batch_y = mnist.test.images,mnist.test.labels
        batch_x = batch_x.reshape(-1,IMAGE_WIDTH,IMAGE_HEIGHT,NUM_CHANNELS)
        batch_y = batch_y.reshape(-1,NUM_CLASS)
        feed_dict={X: batch_x,Y:batch_y,keep_prob:0.8}
        accuracy_val=sess.run(accuracy,feed_dict)
        print "Model-Accuracy: %f" % (accuracy_val)

def main():
    train(data_dir,saved_model_dir,logs_dir)

if __name__=="__main__":
    main()
