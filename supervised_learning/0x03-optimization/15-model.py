#!/usr/bin/env python3
"""Model train"""
import tensorflow as tf
import numpy as np
def create_batch_norm_layer(prev, n, activation):
    """creates a batch normalization layer
    for a neural network in tensorflow"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=None,
                            name='layer',
                            kernel_initializer=init)
    out = layer(prev)
    mean, var = tf.nn.moments(out, axes=0, keep_dims=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    norm = tf.nn.batch_normalization(out, mean, var, offset=beta, scale=gamma,
                                     variance_epsilon=1e-8)
    if not activation:
        return out
    return activation(norm)
def forward_prop(x, layer_sizes=[], activations=[]):
    """Forward Propagation function"""
    x_input = x
    for i in range(len(layer_sizes)):
        x_input = create_batch_norm_layer(x_input, layer_sizes[i], activations[i])
    return x_input
def calculate_accuracy(y, y_pred):
    """ calculates the accuracy of a
        prediction
    - y is a placeholder for the labels
      of the input data
    - y_pred is a tensor containing the
      network's predictions
    """
    correct_prediction = tf.equal(
                                  tf.argmax(y, 1),
                                  tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
def calculate_loss(y, y_pred):
    """calculates the softmax
    cross-entropy loss of a prediction
    - y is a placeholder for the labels
      of the input data
    - y_pred is a tensor containing the
      network's predictions
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """creates the training operation for a neural network
    in tensorflow using the Adam optimization algorithm"""
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return optimizer.minimize(loss)
def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """creates a learning rate decay operation
    in tensorflow using inverse time decay"""
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        g_s = sess.run(global_step)
        if g_s % decay_rate == 0:
            alpha = tf.train.inverse_time_decay(alpha, global_step, decay_step, decay_rate, staircase=True)
            return alpha
        else:
            return alpha
def shuffle_data(X, Y):
    """shuffles the data points
    in two matrices the same way"""
    n = X.shape[0]
    shuffle = np.random.permutation(n)
    return X[shuffle], Y[shuffle]
def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """builds, trains, and saves a neural network model in tensorflow using
    Adam optimization, mini-batch gradient descent, learning rate decay, and
    batch normalization"""
    shape_nx = Data_train[0].shape[1]
    shape_classes = Data_train[1].shape[1]
    x = tf.placeholder(tf.float32, shape=[None, shape_nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, shape_classes], name='y')
    y_pred = forward_prop(x, layers, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 10)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        iterations = Data_train[0].shape[0] // batch_size
        if iterations % batch_size != 0:
            iterations += 1
            flag = True
        else:
            flag = False
        for i in range(epochs + 1):
            sess.run(alpha)
            cost_t, acc_t = sess.run([loss, accuracy], {x: Data_train[0], y: Data_train[1]})
            cost_v, acc_v = sess.run([loss, accuracy], {x: Data_valid[0], y: Data_valid[1]})
            print('After {} epochs:'.format(i))
            print('\tTraining Cost: {}'.format(cost_t))
            print('\tTraining Accuracy: {}'.format(acc_t))
            print('\tValidation Cost: {}'.format(cost_v))
            print('\tValidation Accuracy: {}'.format(acc_v))
            if i < epochs:
                x_t, y_t = shuffle_data(Data_train[0], Data_train[1])
                for j in range(iterations):
                    start = j * batch_size
                    if j == iterations - 1 and flag:
                        end = Data_train[0].shape[0]
                    else:
                        end = j * batch_size + batch_size
                    b_x = x_t[start:end]
                    b_y = y_t[start:end]
                    global_ass = global_step.assign(j)
                    sess.run(global_ass)
                    sess.run(train_op, {x: b_x, y: b_y})
                    if j != 0 and (j + 1) % 100 == 0:
                        new_c, new_ac = sess.run([loss, accuracy], {x: b_x, y: b_y})
                        print('\tStep {}:'.format(j + 1))
                        print('\t\tCost: {}'.format(new_c))
                        print('\t\tAccuracy: {}'.format(new_ac))
        return saver.save(sess, save_path)
