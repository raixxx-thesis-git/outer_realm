import tensorflow as tf

from .apex import Apex
from tensorflow.keras import Model

# untouchable
class OuterRealmMismatch(Exception):
  ''' constructor '''
  def __init__(self, msg):
    super().__init__(msg)

class Assertor():
  ''' 
    DO NOT TOUCH!
    INTERNAL USE ONLY
  '''
  def __init__(self):
    pass

  ''' 
    DO NOT TOUCH!
    INTERNAL USE ONLY
    +----------------------------------------------------------------------+
    description: This method is called to assure the user correctly pass
    data types to the touchable methods'/classes' arguments.
    +----------------------------------------------------------------------+
  '''
  def user_input_assert_type(self, anno: dict, locals_: dict) -> None:
    # used for user-input data types checking
    anno = anno.__annotations__
    for key in anno:
      if type(locals_[key]) != anno[key]:
        raise OuterRealmMismatch((f'User input error. Mismatch data type for key "{key}". Expected {anno[key]} ' 
                                  f'but got {type(locals_[key])}'))
    pass

  ''' 
    DO NOT TOUCH!
    INTERNAL USE ONLY
  '''
  def model_assert_input_compability(self, model: Model, apex_obj: Apex) -> bool:
    pass

  ''' 
    DO NOT TOUCH!
    INTERNAL USE ONLY
  '''
  def model_assert_output_compability(self, model: Model) -> bool:
    pass
