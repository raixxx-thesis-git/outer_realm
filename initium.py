from tensorflow.python.data.ops.unbatch_op import _UnbatchDataset
from tensorflow.python.framework.ops import SymbolicTensor
import tensorflow as tf

from outer_realm.assertor import Assertor

''' 
  * DO NOT TOUCH!
  * INTERNAL USE ONLY
  * Description: This method is called to map the symbolic tensor of the
    dataset reading into a tuple. This method is specifically designed
    to retrieve the data and the epicenter distance information from 'the'
    tfrecords.
'''
def _map_reader(three_channel: bool):
  # complying with the tfrecord structure
  @tf.function
  def map(bin: SymbolicTensor) -> (SymbolicTensor, SymbolicTensor):
    config = {
      'data': tf.io.FixedLenFeature([], tf.string),
      'dist': tf.io.FixedLenFeature([], tf.string)
    }

    # parsing example and tensor
    parsed_example = tf.io.parse_example(bin, config)
    if not three_channel:
      data = tf.io.parse_tensor(parsed_example['data'], tf.float32)[:,0:1,300:5000]
    else:
      data = tf.io.parse_tensor(parsed_example['data'], tf.float32)[:,:,300:5000]

    # transposing BxCxW to BxWxC (B: Batch Size, W: Window Size, C: Channel) 
    data = tf.transpose(data, perm=[0, 2, 1])

    # parsing serialized epicentral distance data
    dist = tf.io.parse_tensor(parsed_example['dist'], tf.float32)
    dist = tf.expand_dims(dist, axis=-1)

    return (data, dist)
  return map

def reader_get_data_and_epicenter(tfrecords_dir: list, three_channel: bool) -> _UnbatchDataset:
  # enforcing the user to comply with the predefined data type
  Assertor().enforce_static_writing(reader_get_data_and_epicenter, locals())

  return tf.data.TFRecordDataset(tfrecords_dir).map(_map_reader(three_channel)).unbatch()
