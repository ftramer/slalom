
import sys
import tensorflow as tf


def _preprocess_conv2d_input_fixed(x, data_format):
    """Transpose and cast the input before the conv2d.

    # Arguments
        x: input tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A tensor.
    """

    tf_data_format = 'NHWC'
    if data_format == 'channels_first':
        if not _has_nchw_support():
            x = tf.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        else:
            tf_data_format = 'NCHW'
    return x, tf_data_format

assert 'keras' not in sys.modules
import keras
 
keras = sys.modules['keras']
keras.backend.tensorflow_backend._preprocess_conv2d_input = _preprocess_conv2d_input_fixed
 
sys.modules['keras'] = keras
