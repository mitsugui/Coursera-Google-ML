import datalab.bigquery as bq
import tensorflow as tf
import numpy as np
import shutil
from google.datalab.ml import TensorBoard

# Prepares input dataset
def create_dataset(mode, batch_size = 512):
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

    dataset = tf.data.Dataset.from_tensor_slices(generate_features_labels(100000))
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        num_epochs = None # indefinitely
        dataset = dataset.shuffle(buffer_size = 10 * batch_size)
    else:
        num_epochs = 1 # end-of-input after this

    dataset = dataset.repeat(num_epochs).batch(batch_size)
    
    return dataset.make_one_shot_iterator().get_next()
  return _input_fn
    

def get_train():
  return create_dataset(mode = tf.estimator.ModeKeys.TRAIN)

def get_valid():
  return create_dataset(mode = tf.estimator.ModeKeys.EVAL)

def get_test():
  return create_dataset(mode = tf.estimator.ModeKeys.EVAL)

INPUT_COLUMNS = [
    tf.feature_column.numeric_column('r'),
    tf.feature_column.numeric_column('h')
]

def add_more_features(feats):
  # Nothing to add (yet!)
  return feats

feature_cols = add_more_features(INPUT_COLUMNS)

#Serving input function
# Defines the expected shape of the JSON feed that the model
# will receive once deployed behind a REST API in production.
def serving_input_fn():
  feature_placeholders = {
    'r' : tf.placeholder(tf.float32, [None]),
    'h' : tf.placeholder(tf.float32, [None])
  }
  # You can transforma data here from the input format to the format expected by your model.
  features = feature_placeholders # no transformation needed
  return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

def train_and_evaluate(output_dir, num_train_steps):
  # estimator = tf.estimator.LinearRegressor(
  #                      model_dir = output_dir,
  #                      feature_columns = feature_cols)
  estimator = tf.estimator.(
                       model_dir = output_dir,
                       feature_columns = feature_cols)
  train_spec=tf.estimator.TrainSpec(
                       input_fn = create_dataset(mode = tf.estimator.ModeKeys.TRAIN),
                       max_steps = num_train_steps)
  exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
  eval_spec=tf.estimator.EvalSpec(
                       input_fn = create_dataset(mode = tf.estimator.ModeKeys.EVAL),
                       steps = None,
                       start_delay_secs = 1, # start evaluating after N seconds
                       throttle_secs = 10,  # evaluate every N seconds
                       exporters = exporter)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


OUTDIR = 'volume_trained'
TensorBoard().start(OUTDIR)

# Run training    
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
train_and_evaluate(OUTDIR, num_train_steps = 2000)

# to list Tensorboard instances
TensorBoard().list()

# to stop TensorBoard fill the correct pid below
TensorBoard().stop(13085)
print("Stopped Tensorboard")