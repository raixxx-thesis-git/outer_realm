from tensorflow.python.data.ops.unbatch_op import _UnbatchDataset
from tensorflow.python.framework.ops import SymbolicTensor
import tensorflow as tf

from .assertor import Assertor

''' 
  * DO NOT TOUCH!
  * INTERNAL USE ONLY
  * Description: This method is called to map the symbolic tensor of the
    dataset reading into a tuple. This method is specifically designed
    to retrieve the data and the epicenter distance information from 'the'
    tfrecords.
'''
@tf.function
def _map_reader(bin: SymbolicTensor) -> (SymbolicTensor, SymbolicTensor):
  # parsing example
  parsed_example = tf.io.parse_example(bin, parsing_config)

  # parsing serialized waveform data
  data = tf.io.parse_tensor(parsed_example['data'], tf.float32)

  # transposing BxCxW to BxWxC (B: Batch Size, W: Window Size, C: Channel) 
  data = tf.transpose(data, perm=[0, 2, 1])

  # parsing serialized epicentral distance data
  dist = tf.io.parse_tensor(parsed_example['dist'], tf.float32)

  return (data, dist)

def reader_get_data_and_epicenter(tfrecords_dir: list) -> _UnbatchDataset:
  # enforcing the user to comply with the predefined data type
  assertor = outer_realm.Assertor()
  assertor.enforce_static_writing(reader_get_data_and_epicenter, locals())

  return tf.data.TFRecordDataset(tfrecords_dir).map(_map_reader).unbatch()
