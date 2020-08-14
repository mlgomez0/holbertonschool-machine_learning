#!/usr/bin/env python3
"""builds, trains, and saves a neural
   network classifier"""
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """returns path where model was saved"""
    x_train, y_train = create_placeholders(X_train.shape[0], Y_train.shape[0])
    x_valid, y_valid = create_placeholders(X_train.shape[0], Y_train.shape[0])
    y_pred = forward_prop(x_train, layer_sizes, activations)
    y_valid = forward_prop(x_valid, layer_sizes, activations)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
            for i in range(iterations + 1):
                at = calculate_accuracy(y_train, y_pred)
                av = calculate_accuracy(Y_valid, y_pred)
                if i % 100 == 0 or i == 0 or i == iterations):
                    print("After {} iterations:\tTraining Cost: {
                          }\tTraining Accuracy: {}\tValidation Cost: {
                          }\tValidation Accuracy: {}".format(i, ))
