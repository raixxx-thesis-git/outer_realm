# from tensorflow.python.data.ops.unbatch_op import _UnbatchDataset
# from tensorflow.python.data.ops.batch_op import _BatchDataset
# from tensorflow.python.data.ops.map_op import _MapDataset
from tensorflow.python.data.ops.unbatch_op import _UnbatchDataset
from tensorflow.python.framework.ops import SymbolicTensor
import tensorflow as tf

from .assertor import Assertor

# untouchable
@tf.function
def _map_reader(bin: SymbolicTensor) -> (SymbolicTensor, SymbolicTensor):
  
  # parsing example
  parsed_example = tf.io.parse_example(bin, parsing_config)

  # parsing tensor
  data = tf.io.parse_tensor(parsed_example['data'], tf.float32)
  data = tf.transpose(data, perm=[0, 2, 1])
  dist = tf.io.parse_tensor(parsed_example['dist'], tf.float32)

  return (data, dist)

# touchable
def reader_get_data_and_epicenter(tfrecords_dir: list) -> _UnbatchDataset:
  
  # restricting data types
  assertor = outer_realm.Assertor()
  assertor.user_input_assert_type(reader_get_data_and_epicenter, locals())

  return tf.data.TFRecordDataset(tfrecords_dir).map(_map_reader).unbatch()
