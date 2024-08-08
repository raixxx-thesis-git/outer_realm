from __future__ import annotations 
from typing import TYPE_CHECKING
from tensorflow.python.framework.ops import EagerTensor

import tensorflow as tf

if TYPE_CHECKING:
    from outer_realm.assertor import Apex

class ApexTrainer():
  def __init__(self, apex: Apex) -> None:
    self.apex = apex
  
  ''' 
    * DO NOT TOUCH! INTERNAL USE ONLY!
    * Description: This method evaluates the model per epoch.
  '''
  @tf.function
  def evaluate_epoch(self, val_data: EagerTensor, expected: EagerTensor) -> EagerTensor:
    # calculations
    predicted = self.apex.model(val_data)
    validation_loss = self.apex.loss(predicted, expected)
    validation_r2 = self.get_r2(predicted, expected)

    return validation_loss, validation_r2

  ''' 
    * DO NOT TOUCH! INTERNAL USE ONLY!
    * Description: This method calculates the R^2 score
  '''
  @tf.function
  def get_r2(self, predicted: EagerTensor, expected: EagerTensor) -> EagerTensor:
    # calculating ss regression: sum of (predicted - expected)^2
    ss_regression = tf.math.reduce_sum(tf.math.square(predicted - expected))

    # calculating ss total: sum of (predicted - mean of expected)^2
    ss_total = tf.math.reduce_sum(tf.math.square(expected - tf.math.reduce_mean(expected)))

    return 1 - (ss_regression / ss_total)

  ''' 
    * DO NOT TOUCH! INTERNAL USE ONLY!
    * Description: This method is called to update model's tensors with 
      gradient updating via computational graph backward propagation.
  '''
  @tf.function
  def update_trainable_tensors(self, train_data: EagerTensor, expected: EagerTensor) -> EagerTensor:
    # applying backward propagation gradient
    with tf.GradientTape() as d:
      # forward propagation: predicting value and calculating loss
      predicted = self.apex.model(train_data)
      training_loss = self.apex.loss(predicted, expected)

      # calculating ∂L/∂θ and applying θ := θ - α(∂L/∂θ)
      grad = d.gradient(training_loss, self.apex.model.trainable_variables)
      self.apex.temp_optimizer.apply_gradients(zip(grad, self.apex.model.trainable_variables))
    
    return training_loss