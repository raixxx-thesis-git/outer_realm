from __future__ import annotations 
from typing import TYPE_CHECKING
from keras.src.models.functional import Functional

import tensorflow as tf

if TYPE_CHECKING:
    from outer_realm.assertor import Apex

''' 
  * DO NOT TOUCH! INTERNAL USE ONLY!
'''
class OuterRealmMismatch(Exception):
  ''' constructor '''
  def __init__(self, msg):
    super().__init__(msg)

''' 
  * DO NOT TOUCH! INTERNAL USE ONLY!
'''
class Assertor():
  def __init__(self):
    pass

  ''' 
  * Description: This method is called to assure the user correctly pass
    data types to the touchable methods'/classes' arguments.
  '''
  def enforce_static_writing(self, anno: dict, locals_: dict, exceptions: list = []) -> None:
    # used for user-input data types checking
    anno = anno.__annotations__
    for key in locals_:
      if key == 'self': continue
      if key in exceptions: continue
      if type(locals_[key]) != anno[key]:
        raise OuterRealmMismatch((f'User input error. Mismatch data type for key "{key}". Expected {anno[key]} ' 
                                  f'but got {type(locals_[key])}'))
    pass

  def model_assert_input_compability(self, model: Functional, apex_obj: Apex) -> None:
    # expected window length: W
    expected_window_length = apex_obj.window_length

    # expected channel size: C
    expected_channel = apex_obj.channel_size

    model_input_shape = model.layers[0].output.shape

    # model should be (None, W, C) 
    if len(model_input_shape) != 3:
      raise OuterRealmMismatch(f'Model error. Expected (None, {expected_window_length}, {expected_channel}) '
                               f'but got {model_input_shape}')

    # model's window length != W
    if model_input_shape[1] != expected_window_length:
      raise OuterRealmMismatch((f'Model is incompatible with the expected window length!' 
                                f' Your model expects {model_input_shape[1]}'
                                f' but {expected_window_length[1]} is expected!'))

    # model's channel size != C
    if model_input_shape[2] != expected_channel:
      raise OuterRealmMismatch((f'Model is incompatible with the expected channel size! Your model expects {model_input_shape[1]}'
                                f' but {expected_window_length[1]} is expected!'))
    pass

  def model_assert_output_compability(self, model: Functional, apex_obj: Apex) -> None:
    expected_output = 1
    model_output_shape = model.layers[-1].output.shape

    if len(model_output_shape) != 2:
      raise OuterRealmMismatch(f'Model error. Expected (None, 1) but got {model_output_shape}')

    # model's output is not a scalar (or a vector if the batch is considered)
    if model_output_shape[1] != expected_output:
      raise OuterRealmMismatch((f'Model is incompatible with the output!' 
                                f' Your model expects {model_output_shape[1]}'
                                f' but {expected_output} is expected!'))
    pass
