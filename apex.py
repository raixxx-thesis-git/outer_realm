'''
Everything that pertains to model intiation, model training, model updating, model evaluation is provided in here.
'''

from tensorflow.python.eager.polymorphic_function.polymorphic_function import Function
from tensorflow.python.data.ops.unbatch_op import _UnbatchDataset
from tensorflow.python.data.ops.batch_op import _BatchDataset
from tensorflow.python.framework.ops import EagerTensor
from keras.src.models.functional import Functional
from outer_realm import Assertor
from outer_realm import ApexTrainer
from tensorflow.keras import Model
from tqdm.std import tqdm as Tqdm

import tensorflow as tf
import outer_realm
import tqdm
import json
import copy
import uuid
import gc
import os

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
    print('Please wait. Checking model compability.')
    self.model_assert_input_compability(model, self)
    self.model_assert_output_compability(model, self)
    print('Done! Your model is compatible.')
    
    # check dataset compability
    print('Please wait. Checking dataset compability.')
    self.dataset_assert_compability(training_dataset, 'training', self)
    self.dataset_assert_compability(validation_dataset, 'validation', self)
    print('Done! Your dataset is compatible.')

    # assign model and dataset if compatible
    self.training_dataset = training_dataset
    self.validation_dataset = validation_dataset
    self.model = model

    # optional(s):
    self.total_batch = None

  ''' 
    * Description: This method is used to assign an optimizer (MANDATORY!)
  '''
  def update_optimizer(self, optimizer: any) -> None:
    self.optimizer = optimizer

  ''' 
    * Description: This method is used to update the user's model
  '''
  def update_model(self, model: Functional) -> None:
    self.enforce_static_writing(self.update_model, locals())

    # check model compability
    print('Please wait. Checking model compability.')
    self.model_assert_input_compability(model, self)
    self.model_assert_output_compability(model, self)
    print('Done! Your model is compatible.')

    # only if the model passes all the test the model would be updated
    self.model = model

  ''' 
    * Description: This method is used to update the user's datasets
  '''
  def update_dataset(self, training_dataset: _UnbatchDataset, validation_dataset: _UnbatchDataset) -> None:
    self.enforce_static_writing(self.update_dataset, locals())

    # check dataset compability
    print('Please wait. Checking dataset compability.')
    self.dataset_assert_compability(training_dataset, 'training', self)
    self.dataset_assert_compability(validation_dataset, 'validation', self)
    print('Done! Your dataset is compatible.')

    # update dataset
    self.training_dataset = training_dataset
    self.validation_dataset = validation_dataset

  ''' 
    * Description: This method updates the batch size
  '''
  def update_batch_size(self, batch_size: int) -> None:
    # enforce the user to comply with the predefined data type
    self.enforce_static_writing(self.update_batch_size, locals())

    self.batch_size = batch_size

  ''' 
    * Description: This method updates the total epoch
  '''
  def update_epoch(self, epoch: int) -> None:
    # enforce the user to comply with the predefined data type
    self.enforce_static_writing(self.update_epoch, locals())

    self.epoch = epoch

  ''' 
    * Description: This method updates the window length configuration.
  '''
  def update_window_length(self, window_length: int) -> None:
    # enforce the user to comply with the predefined data type
    self.enforce_static_writing(self.update_window_length, locals())

    self.window_length = window_length

  ''' 
    * Description: This method updates the channel size configuration.
  '''
  def update_channel_size(self, channel_size: int) -> None:
    # enforce the user to comply with the predefined data type
    self.enforce_static_writing(self.update_channel_size, locals())

    self.channel_size = channel_size

  ''' 
    * Description: This method would replace the default loss function (MSE)
    with your loss function (user defined loss).
  '''
  def update_loss(self, loss: Function) -> None:
    # enforce the user to comply with the predefined data type
    self.enforce_static_writing(self.update_loss, locals())

    self.loss = loss    

  ''' 
    * Description: This method clears previous keras backend's sessions and 
    collects garbages.
  '''
  def memory_refresh(self) -> None:
    tf.keras.backend.clear_session()
    gc.collect()

  ''' 
    * Description: This method is used to save your model.
  '''
  def save_model(self, path: str) -> None:
    # enforce the user to comply with the predefined data type
    self.enforce_static_writing(self.save_model, locals())

    self.model.save(path)

  ''' 
    * Description: This method is called to train the model.
  '''
  def train(self, save_model_per_epoch: bool=False, calculate_r2_per_epoch: bool=True) -> None:
    # enforce the user to comply with the predefined data type
    self.enforce_static_writing(self.train, locals())

    # generate unique training id
    self.training_session_id = str(uuid.uuid4())
    os.mkdir(self.training_session_id)
    print((f'Your training session ID: {self.training_session_id}.\nAll training logs and model will ' 
           f'automatically be saved in {self.training_session_id} folder.'))

    # refresh memory
    self.memory_refresh()

    # initiating a trainer
    self.apex_trainer = ApexTrainer(self)

    # create a copy of an optimizer
    self.temp_optimizer = copy.deepcopy(self.optimizer)

    # count total batch
    self.total_batch = 0

    # dataset & logs
    train_dataset = self.training_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = self.validation_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
    logs = {'id': self.training_session_id, 
            'training_loss':[], 
            'validation_loss':[], 
            'validation_r2':[]}

    # start training
    print('\nEntering training stage now.\nNote: Proper progress bar appears at the second epoch.\n')

    for epoch in range(1, self.epoch + 1):
      # logging
      print(f'Epoch {epoch}/{self.epoch}')

      # draw training bar, appears at the second epoch
      bar = self.draw_training_bar(self.total_batch)

      # updating model's tensor (learning)
      for i, train_data in enumerate(train_dataset.take(-1)):
        # info: the following call is optimized.
        training_loss = float(self.apex_trainer.update_trainable_tensors(train_data[0], train_data[1]))

        # updating logs
        bar.set_description_str(f'Batch {i}/{self.total_batch} | Training Loss: {training_loss:.4f}')
        bar.update(1)

        # counting up the batch
        if epoch == 1:
          self.total_batch += 1

      bar.close()

      # if model save per epoch only
      if save_model_per_epoch:
        self.model.save(f'{self.training_session_id}/model-epoch-{epoch}.keras')
    
      logs['training_loss'].append(training_loss)
      # validation evaluation
      validation_loss = []
      validation_r2 = []

      for val_data in val_dataset.take(-1):
        validation_loss.append(self.apex_trainer.evaluate_epoch(val_data[0], val_data[1])[0])
        if calculate_r2_per_epoch:
          validation_r2.append(self.apex_trainer.evaluate_epoch(val_data[0], val_data[1])[1])

      # converting into a tensor
      validation_loss = tf.convert_to_tensor(validation_loss)
      validation_loss = float(tf.math.reduce_mean(validation_loss))
      logs['validation_loss'].append(validation_loss)

      print(f'Validation Loss: {validation_loss:.4f}')
      if calculate_r2_per_epoch:
        validation_r2.append(self.apex_trainer.evaluate_epoch(val_data[0], val_data[1])[1])
        validation_r2 = float(tf.math.reduce_mean(validation_r2))
        print(f'Validation R2: {validation_r2:.4f}%\n')
        logs['validation_r2'].append(validation_r2)


    self.write_json(logs)
    del self.apex_trainer
    self.memory_refresh()
    self.model.save(f'{self.training_session_id}/model-end.keras')
    print('Closed training session, Apex Trainer is freed.')

  ''' 
    * DO NOT TOUCH! INTERNAL USE ONLY!
    * Description: This method writes the model training history in json.
  '''
  def write_json(self, logs: dict) -> None:
    json_object = json.dumps(logs)
    with open(f'{self.training_session_id}/metadata.json', 'w') as f:
      f.write(json_object)
  
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
    * Description: This method draws a training progress bar.
  '''
  def draw_training_bar(self, total_batch: int) -> Tqdm:
    return tqdm.tqdm(total=total_batch, ascii='._█', position=0, bar_format='|{bar:30}| [{elapsed}<{remaining}] {desc}')

  # ''' 
  #   * DO NOT TOUCH! INTERNAL USE ONLY!
  #   * Description: This method calculates how many batch exists in a batch dataset.
  # '''
  # @tf.function
  # def get_dataset_length(self, dataset: _BatchDataset) -> int:
  #   return dataset.reduce(0, lambda x,_: x+1)