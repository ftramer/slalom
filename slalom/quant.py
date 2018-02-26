import tensorflow as tf
from keras.layers import Conv2D, Dense
import keras.backend as K

QUANTIZE = True
PRINT = False

class Conv2DQ(Conv2D):
    def __init__(self, filters, kernel_size, bits_w=8, bits_x=8, **kwargs):
        super(Conv2DQ, self).__init__(filters, kernel_size, **kwargs)
        self.quantize = QUANTIZE
        self.bits_w = bits_w
        self.bits_x = bits_x
        self.range_w = 2**bits_w
        self.range_x = 2**bits_x
        self.kernel_q = None
        self.bias_q = None

    def build(self, input_shape):
        super(Conv2DQ, self).build(input_shape)
        if self.quantize:
            self.kernel_q = K.round(self.range_w * self.kernel)
            if self.use_bias:
                self.bias_q = K.round(self.range_w * self.range_x * self.bias)

        else:
            self.kernel_q = self.kernel
            if self.use_bias:
                self.bias_q = self.bias

    def call(self, inputs):

        outputs = K.conv2d(
            inputs,
            self.kernel_q,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if PRINT:
            outputs = tf.Print(outputs, [tf.reduce_max(tf.abs(outputs))])

        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias_q, data_format=self.data_format)

        if self.activation is not None:
            outputs = self.activation(outputs)

        if self.quantize:
            return K.round(outputs / self.range_w)
        else:
            return outputs

    def get_config(self):
        config = {
            'bits_w': self.bits_w,
            'bits_x': self.bits_x
        }
        
        base_config = super(Conv2DQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DenseQ(Dense):

    def __init__(self, units, bits_w=8, bits_x=8, **kwargs):
        super(DenseQ, self).__init__(units, **kwargs)

        self.quantize = QUANTIZE
        self.bits_w = bits_w
        self.bits_x = bits_x
        self.range_w = 2**bits_w
        self.range_x = 2**bits_x
        self.kernel_q = None
        self.bias_q = None

    def build(self, input_shape):
        super(DenseQ, self).build(input_shape)
        if self.quantize:
            self.kernel_q = K.round(self.range_w * self.kernel)
            if self.use_bias:
                self.bias_q = K.round(self.range_w * self.range_x * self.bias)

        else:
            self.kernel_q = self.kernel
            if self.use_bias:
                self.bias_q = self.bias

    def call(self, inputs, blind=True, bias=True):

        outputs = K.dot(inputs, self.kernel_q)

        if PRINT:
            outputs = tf.Print(outputs, [tf.reduce_max(tf.abs(outputs))])

        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias_q)

        if self.activation is not None:
            outputs = self.activation(outputs)

        if self.quantize:
            return K.round(outputs / self.range_w)
        else:
            return outputs

    def get_config(self):
        config = {
            'bits_w': self.bits_w,
            'bits_x': self.bits_x
        }

        base_config = super(DenseQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

