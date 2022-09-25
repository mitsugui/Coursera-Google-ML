from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import shutil

tf.logging.set_verbosity(tf.logging.INFO)

# Prepares input dataset
def create_dataset(mode, batch_size = 512, data_size = 1000000):
  def _input_fn():    
    def random_values(size):
      return 0.5 + 1.5 * np.random.rand(size)
    
    def calc_volume(r, h):
      return np.pi * r * r * h
    
    def add_noise(original):
      return np.around(original + 0.1*np.random.rand(original.size), decimals=1)
    
    def generate_features_labels(size):
      r = random_values(size)
      h = random_values(size)
      noisy_r = add_noise(r)
      noisy_h = add_noise(h)
      features = {
        'r': noisy_r,
        'h': noisy_h
      }
      label = add_noise(calc_volume(r, h))
      return features, label

    dataset = tf.data.Dataset.from_tensor_slices(generate_features_labels(data_size))
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        num_epochs = None # indefinitely
        dataset = dataset.shuffle(buffer_size = 10 * batch_size)
    else:
        num_epochs = 1 # end-of-input after this

    dataset = dataset.repeat(num_epochs).batch(batch_size)
    
    return dataset.make_one_shot_iterator().get_next()
  return _input_fn

INPUT_COLUMNS = [
    tf.feature_column.numeric_column('r'),
    tf.feature_column.numeric_column('h')
]

# Create a function that will augment your feature set
def add_more_features(feats):
    # Nothing to add (yet!)
    return feats

feature_cols = add_more_features(INPUT_COLUMNS)

# Create your serving input function so that your trained model will be able to serve predictions
def serving_input_fn():
    feature_placeholders = {
        column.name: tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS
    }
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

def train_and_evaluate(args):
    estimator = tf.estimator.DNNRegressor(
                        model_dir = args['output_dir'],
                        feature_columns = feature_cols,
                        hidden_units = args['hidden_units'])
    train_spec=tf.estimator.TrainSpec(
                        input_fn = create_dataset(mode = tf.estimator.ModeKeys.TRAIN, 
                                                  batch_size = args['train_batch_size'],
                                                  data_size = args['train_data_size']),
                        max_steps = args['train_steps'])
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec=tf.estimator.EvalSpec(
                        input_fn = create_dataset(mode = tf.estimator.ModeKeys.EVAL, 
                                                  data_size = args['eval_data_size']),
                        steps = None,
                        start_delay_secs = args['eval_delay_secs'], # start evaluating after N seconds
                        throttle_secs = args['min_eval_frequency'],  # evaluate every N seconds
                        exporters = exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
