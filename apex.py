'''
Everything that pertains to model intiation, model training, model updating, model evaluation is provided in here.
'''
import tensorflow as tf
import outer_realm

from tensorflow.python.eager.polymorphic_function.polymorphic_function import Function
from tensorflow.python.data.ops.unbatch_op import _UnbatchDataset
from tensorflow.python.framework.ops import EagerTensor
from outer_realm import Assertor
from tensorflow.keras import Model

class Apex(Assertor):
  def __init__(self,
               training_dataset: _UnbatchDataset,
               validation_dataset: _UnbatchDataset,
               batch_size: int,
               window_length: int,
               model: Model,
               user_loss: Function = None,
               ) -> None:
    # enforcing the user to comply with the predefined data type
    self.enforce_static_writing(self.__init__, locals())

    # configurations
    self.training_dataset = training_dataset
    self.validation_dataset = validation_dataset
    self.batch_size = batch_size
    self.window_length = window_length
    self.user_loss = user_loss
    
    # the tests: check model compability
    self.model_check_compability()

    # adding model to the object (only works after it passes the tests)
    self.model = model

  def model_check_compability():
    assertor.model_assert_input_compability(self.model, self)
    assertor.model_assert_output_compability(self.model, self)

  def define_optimizer(self, optimizer):
    self.optimizer = optimizer

  def update_model(self, model: Model) -> None:
    assertor.model_assert_input_compability(self.model, self)
    assertor.model_assert_output_compability(self.model, self)
    
  ''' 
    * DO NOT TOUCH! INTERNAL USE ONLY!
    * Description: This method is used to calculate the loss of the model
      during the training session.
  '''
  @tf.function
  def loss(predicted: EagerTensor, expected: EagerTensor) -> EagerTensor:
    # use templatic loss if user loss is not defined
    if self.user_loss == None:
      sum_squared = tf.math.mean((predicted - expected)**2, axis=0, keepdims=False)
      return sum_squared[0]

    # use user loss if user loss is defined
    else:
      return self.user_loss(predicted, expected)

  ''' 
    * DO NOT TOUCH! INTERNAL USE ONLY!
    * Description: This method is called to update model's tensors with 
      gradient updating via computational graph backward propagation.
  '''
  @tf.function
  def backprop_apply_gradient(predicted: EagerTensor, expected: EagerTensor) -> EagerTensor:
    # applying backward propagation gradient
    with tf.GradientTaping() as d:
      # calculating loss
      training_loss = self.user_loss(predicted, expected)

      # calculating ∂L/∂θ
      grad = d.gradient(training_loss, self.model.trainable_variables)

      # updating θ := θ - α(∂L/∂θ)
      self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
    
    return training_loss

    