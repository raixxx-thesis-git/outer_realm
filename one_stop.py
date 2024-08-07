'''
Everything that pertains to model intiation, model training, model updating, model evaluation is provided in here.
'''
import tensorflow as tf

class OneStop():
  def __init__(self,
               training_dataset,
               validation_dataset,
               batch_size,
               window_length,
               ):

    self.training_dataset = training_dataset
    self.validation_dataset = validation_dataset
    self.batch_size = batch_size
    self.window_length = window_length