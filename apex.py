'''
Everything that pertains to model intiation, model training, model updating, model evaluation is provided in here.
'''

from tensorflow.python.eager.polymorphic_function.polymorphic_function import Function
from tensorflow.python.data.ops.unbatch_op import _UnbatchDataset
from tensorflow.python.data.ops.batch_op import _BatchDataset
from tensorflow.python.framework.ops import EagerTensor
from keras.src.models.functional import Functional
from outer_realm import Assertor
from tensorflow.keras import Model
from tqdm.std import tqdm as Tqdm

import tensorflow as tf
import outer_realm
import tqdm
import copy
import gc

class Apex(Assertor):
  def __init__(self,
               training_dataset: _UnbatchDataset,
               validation_dataset: _UnbatchDataset,
               window_length: int,
               channel_size: int,
               batch_size: int,
               model: Functional,
               epoch: int,
               user_loss: Function = None,
               ) -> None:
    # enforcing the user to comply with the predefined data type
    self.enforce_static_writing(self.__init__, locals(), exceptions=['user_loss'])

    # configurations
    self.batch_size = batch_size
    self.training_dataset = training_dataset.batch(batch_size)
    self.validation_dataset = validation_dataset.batch(batch_size)
    self.window_length = window_length
    self.channel_size = channel_size
    self.user_loss = user_loss
    self.epoch = epoch
    
    # the tests: check model compability
    self.model_check_compability(model)

    # adding model to the object (only works after it passes the tests)
    self.model = model


  def define_optimizer(self, optimizer):
    self.optimizer = optimizer

  def update_model(self, model: Functional) -> None:
    # enforcing the user to comply with the predefined data type
    self.enforce_static_writing(self.update_model, locals())

    # check model compability
    self.model_check_compability(model)

    # only if the model passes all the test the model would be updated
    self.model = model

  def train(self):
    # refreshing memory
    tf.keras.backend.clear_session()
    gc.collect()

    # creating a copy optimizer
    self.temp_optimizer = copy.deepcopy(self.optimizer)

    # count total batch
    total_batch = get_dataset_length(self.training_dataset)

    # dataset
    train_dataset = self.training_dataset.take(-1)
    val_dataset = self.validation_dataset

    # start training
    for epoch in range(1, self.epoch + 1):
      # draw training bar
      bar = draw_training_bar(total_batch)

      # updating model's tensor (learning)
      for train_data in train_dataset:
        # info: the following call is optimized.
        training_loss = float(self.update_trainable_tensors(train_data[0], train_data[1]))

        bar.set_description_str(f'Training Loss: {training_loss:.4f}')
        bar.update(1)
      
      # close bar after an epoch
      bar.close()
    
      # validation eval
      validation_loss = tf.constant(0.0)

      for val_data in val_dataset:
        predicted = self.model(val_data[0])
        expected = val_data[1]
        validation_loss += self.loss(predicted, expected)

      # average validation loss
      validation_loss = float(tf.math.reduce_mean(validation_loss))

      print(f'Validation Loss: {validation_loss:.4f}')

  ''' 
    * DO NOT TOUCH! INTERNAL USE ONLY!
    * Description: This method checks whether the model complies with the
      user's defined configuration.
  '''
  def model_check_compability(self, model):
    self.model_assert_input_compability(model, self)
    self.model_assert_output_compability(model, self)
    
  ''' 
    * DO NOT TOUCH! INTERNAL USE ONLY!
    * Description: This method is used to calculate the loss of the model
      during the training session.
  '''
  @tf.function
  def loss(self, predicted: EagerTensor, expected: EagerTensor) -> EagerTensor:
    # use templatic loss if user loss is not defined
    if self.user_loss == None:
      sum_squared = tf.math.mean((predicted - expected)**2, axis=0, keepdims=False)
      return sum_squared[0]

    # use user loss if user loss is defined
    else:
      # user loss must return a scalar!
      return self.user_loss(predicted, expected)

  ''' 
    * DO NOT TOUCH! INTERNAL USE ONLY!
    * Description: This method is called to update model's tensors with 
      gradient updating via computational graph backward propagation.
  '''
  @tf.function
  def update_trainable_tensors(self, train_data: EagerTensor, expected: EagerTensor) -> EagerTensor:
    # forward propagation: predicting value
    predicted = self.model(train_data)

    # applying backward propagation gradient
    with tf.GradientTaping() as d:
      # calculating loss
      training_loss = self.user_loss(predicted, expected)

      # calculating ∂L/∂θ
      grad = d.gradient(training_loss, self.model.trainable_variables)

      # updating θ := θ - α(∂L/∂θ)
      self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
    
    return training_loss

  ''' 
    * DO NOT TOUCH! INTERNAL USE ONLY!
    * Description: This method calculates how many batch exists in a batch dataset.
  '''
  @tf.function
  def get_dataset_length(self, dataset: _BatchDataset) -> int:
    return dataset.reduce(0, lambda x,_: x+1)

  ''' 
    * DO NOT TOUCH! INTERNAL USE ONLY!
    * Description: This method draws a training progress bar.
  '''
  def draw_training_bar(self, total_batch: int) -> Tqdm:
    return tqdm.tqdm(total=total_batches, ascii='._█', position=0, bar_format='|{bar:30}| [{elapsed}<{remaining}] {desc}')
