#!/usr/bin/env python
"""evaluates based on a existing model"""
import tensorflow as tf
import numpy as np

def evaluate(X, Y, save_path):
    """
    returns prediction, accuracy and loss_val
    """
    #train_op = tf.get_collection("train_op")
    

    saver = tf.train.import_meta_graph('model.ckpt.meta')
    with tf.Session() as sess:
        #saver.restore(sess, save_path)
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")
        loss = tf.get_collection("loss")
        accuracy = tf.get_collection("accuracy")
        prediction, accuracy_val, loss_val = sess.run(fetches=[y_pred, accuracy, loss], feed_dict={x: X, y: Y})
        return (prediction[0], accuracy_val[0], loss_val[0])
