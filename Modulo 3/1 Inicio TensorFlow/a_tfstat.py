import tensorflow as tf
import numpy as np

def f_x(a, x):
  return a[0] + a[1] * x + a[2] * tf.pow(x, 2) + a[3] * tf.pow(x, 3) + a[4] * tf.pow(x, 4)

def f1_x(a, x):
  return a[1] + 2 * a[2] * x + 3 * a[3] * tf.pow(x, 2) + 4 * a[4] * tf.pow(x, 3)

def f2_x(a, x):
  return 2 * a[2] + 6 * a[3] * x + 12 * a[4] * tf.pow(x, 2)
  
def compute_x_n_plus_1(a, xn):
  return xn-((2*f_x(a,xn)*f1_x(a,xn))/(2*f1_x(a,xn)*f1_x(a,xn)-f_x(a,xn)*f2_x(a,xn)))

with tf.Session() as sess:
  a = tf.constant([0.0, 0.0, 2.0, 3.0, 1.0])
  x0 = tf.constant([-0.001])
    
  c = lambda a, xn, xn_1: tf.squeeze(tf.abs(xn_1-xn)) > 0.0001
  b = lambda a, xn, xn_1: (a, xn_1, compute_x_n_plus_1(a, xn_1))
  
  y = tf.while_loop(c, b, (a, x0, x0 + [-2.0]))
  
  result = sess.run(y)
  print result