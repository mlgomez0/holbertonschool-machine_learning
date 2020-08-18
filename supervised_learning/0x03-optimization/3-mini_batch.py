#!/usr/bin/env python3

import numpy as np
import tensorflow as tf


shuffle_data = __import__('2-shuffle_data').shuffle_data

def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
       
    with tf.Session() as sess:
        load_p = load_path + ".meta"
        saver_n = tf.train.import_meta_graph(load_p)
        saver_n.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        train_op = tf.get_collection("train_op")
        loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]

        step_next = X_train.shape[0] // batch_size
        if step_next % batch_size != 0:
            step_next = step_next + 1
            flag = True
        else:
            flag = False

        for i in range(epochs + 1):
            cost_t, acc_t = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            cost_v, acc_v = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_t))
            print("\tTraining Accuracy: {}".format(acc_t))
            print("\tValidation Cost: {}".format(cost_v))
            print("\tValidation Accuracy: {}".format(acc_v))
            if i < epochs:
                x_train_s, y_train_s = shuffle_data(X_train, Y_train)
                for j in range(step_next):
                    start = j * batch_size
                    if j == step_next - 1 and flag == True:
                        end = X_train.shape[0]
                    else:
                        end = j * batch_size + batch_size
                    batch_X = x_train_s[start:end]
                    batch_Y = y_train_s[start:end]
                    sess.run([train_op], feed_dict={x: batch_X, y: batch_Y})         
                    if (j + 1) % 100 == 0 and j != 0:
                        cost_b, acc_b = sess.run([loss, accuracy], feed_dict={x: batch_X, y: batch_Y})
                        print("\tStep {}".format(j + 1))
                        print("\t\tCost: {}".format(cost_b))
                        print("\t\tAccuracy: {}".format(acc_b))
                    
        return saver_n.save(sess, save_path) 
