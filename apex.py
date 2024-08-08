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
    # enforce the user to comply with the predefined data type
    self.enforce_static_writing(self.__init__, locals(), exceptions=['user_loss'])

    # configurations
    self.batch_size = batch_size
    self.window_length = window_length
    self.channel_size = channel_size
    self.user_loss = user_loss
    self.epoch = epoch
    self.loss = self.default_loss if user_loss == None else user_loss

    # check model compability
    self.model_check_compability(model)
    
    # check dataset compability
    # self.dataset_assert_compability(training_dataset, 'training', self)
    # self.dataset_assert_compability(validation_dataset, 'validation', self)

    # assign model and dataset if compatible
    self.training_dataset = training_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    self.validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    self.model = model

    # optional(s):
    self.total_batch = None

  def update_optimizer(self, optimizer: any) -> None:
    self.optimizer = optimizer

  def update_model(self, model: Functional) -> None:
    self.enforce_static_writing(self.update_model, locals())

    # check model compability
    self.model_check_compability(model)

    # only if the model passes all the test the model would be updated
    self.model = model

  def update_dataset(self, training_dataset: _UnbatchDataset, validation_dataset: _UnbatchDataset) -> None:
    self.enforce_static_writing(self.update_dataset, locals())

    # check dataset compability
    self.dataset_assert_compability(training_dataset, 'training', self)
    self.dataset_assert_compability(validation_dataset, 'validation', self)

    # update dataset
    self.training_dataset = training_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    self.validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  def update_batch_size(self, batch_size: int) -> None:
    # enforce the user to comply with the predefined data type
    self.enforce_static_writing(self.update_batch_size, locals())

    self.batch_size = batch_size

  def update_epoch(self, epoch: int) -> None:
    # enforce the user to comply with the predefined data type
    self.enforce_static_writing(self.update_epoch, locals())

    self.epoch = epoch

  def update_window_length(self, window_length: int) -> None:
    # enforce the user to comply with the predefined data type
    self.enforce_static_writing(self.update_window_length, locals())

    self.window_length = window_length

  def update_channel_size(self, channel_size: int) -> None:
    # enforce the user to comply with the predefined data type
    self.enforce_static_writing(self.update_channel_size, locals())

    self.channel_size = channel_size

  def train(self):
    # refresh memory
    tf.keras.backend.clear_session()
    gc.collect()

    # create a copy of an optimizer
    self.temp_optimizer = copy.deepcopy(self.optimizer)

    # count total batch
    self.total_batch = 0

    # dataset
    train_dataset = self.training_dataset.take(-1)
    val_dataset = self.validation_dataset.take(-1)

    # start training
    print('Entering training stage now.\nNote: The proper progress bar appears at the second epoch.')

    for epoch in range(1, self.epoch + 1):
      #logging
      print(f'Epoch {epoch}/{self.epoch}')

      # draw training bar, appears at the second epoch
      bar = self.draw_training_bar(self.total_batch)

      # updating model's tensor (learning)
      for train_data in train_dataset:
        # info: the following call is optimized.
        training_loss = float(self.update_trainable_tensors(train_data[0], train_data[1]))

        # updating logs
        bar.set_description_str(f'Training Loss: {training_loss:.4f}')
        bar.update(1)

        if epoch == 1:
          # counting up the batch
          self.total_batch += 1
      
      # bar appears at epoch = 2
      bar.close()
    
      # validation eval
      validation_loss = []
      validation_r2 = []

      for val_data in val_dataset:
        validation_loss.append(self.evaluate_epoch(val_data[0], val_data[1])[0])
        validation_r2.append(self.evaluate_epoch(val_data[0], val_data[1])[1])

      # converting into a tensor
      validation_loss = tf.convert_to_tensor(validation_loss)
      validation_r2 = tf.convert_to_tensor(validation_r2)

      # average validation loss
      validation_loss = float(tf.math.reduce_mean(validation_loss))
      validation_r2 = float(tf.math.reduce_mean(validation_r2))

      print(f'Validation Loss: {validation_loss:.4f}')
      print(f'Validation R2: {validation_r2:.4f}%\n')

  ''' 
    * DO NOT TOUCH! INTERNAL USE ONLY!
    * Description: This method evaluates the model per epoch.
  '''
  @tf.function
  def evaluate_epoch(self, val_data: EagerTensor, expected: EagerTensor) -> EagerTensor:
    predicted = self.model(val_data)
    validation_loss = self.loss(predicted, expected)
    validation_r2 = self.get_r2(predicted, expected)
    return validation_loss, validation_r2

  ''' 
    * DO NOT TOUCH! INTERNAL USE ONLY!
    * Description: This method calculates the R^2 score
  '''
  @tf.function
  def get_r2(self, predicted: EagerTensor, expected: EagerTensor) -> EagerTensor:
    ss_regression = tf.math.reduce_sum(tf.math.square(predicted - expected))
    ss_total = tf.math.reduce_sum(tf.math.square(expected - tf.math.reduce_mean(expected)))
    return 1 - (ss_regression / ss_total)

  ''' 
    * DO NOT TOUCH! INTERNAL USE ONLY!
    * Description: This method checks whether the model complies with the
      user's defined configuration.
  '''
  def model_check_compability(self, model: Functional):
    self.model_assert_input_compability(model, self)
    self.model_assert_output_compability(model, self)
    
  ''' 
    * DO NOT TOUCH! INTERNAL USE ONLY!
    * Description: This method is used to calculate the loss of the model
      during the training session.
  '''
  @tf.function
  def default_loss(self, predicted: EagerTensor, expected: EagerTensor) -> EagerTensor:
    # use default loss if user loss is not defined: an MSE loss
    squared_vec = tf.math.square(predicted - expected)
    sum_squared = tf.math.reduce_mean(squared_vec, axis=0, keepdims=False)
    return sum_squared[0]

  ''' 
    * DO NOT TOUCH! INTERNAL USE ONLY!
    * Description: This method is called to update model's tensors with 
      gradient updating via computational graph backward propagation.
  '''
  @tf.function
  def update_trainable_tensors(self, train_data: EagerTensor, expected: EagerTensor) -> EagerTensor:

    # applying backward propagation gradient
    with tf.GradientTape() as d:
      # forward propagation: predicting value
      predicted = self.model(train_data)

      # calculating loss
      training_loss = self.loss(predicted, expected)

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
    return tqdm.tqdm(total=total_batch, ascii='._█', position=0, bar_format='|{bar:30}| [{elapsed}<{remaining}] {desc}')
