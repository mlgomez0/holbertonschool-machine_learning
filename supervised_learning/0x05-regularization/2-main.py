#!/usr/bin/env python3

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_loss = __import__('4-calculate_loss').calculate_loss
regularization_loss = __import__('2-l2_reg_cost').l2_reg_cost

x, y = create_placeholders(784, 10)
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
loss = calculate_loss(y, y_pred)
regu_loss = regularization_loss(loss)
print(loss)
print(regu_loss)