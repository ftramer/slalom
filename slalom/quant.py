from keras.layers import Conv2D, Dense
import keras.backend as K


class Conv2DQ(Conv2D):
    def __init__(self, filters, kernel_size, bits_w=8, bits_x=8, **kwargs):
        super(Conv2DQ, self).__init__(filters, kernel_size, **kwargs)

        self.range_w = 2**bits_w
        self.range_x = 2**bits_x
        self.kernel_q = None
        self.bias_q = None

    def build(self, input_shape):
        super(Conv2DQ, self).build(input_shape)
        self.kernel_q = K.round(self.range_w * self.kernel)
        self.bias_q = K.round(self.range_w * self.range_x * self.bias)

    def call(self, inputs, blind=True, bias=True):

        outputs = K.conv2d(
            inputs,
            self.kernel_q,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if bias:
            outputs = K.bias_add(outputs, self.bias_q, data_format=self.data_format)

        return K.round(outputs / self.range_w)


class DenseQ(Dense):

    def __init__(self, units, bits_w=8, bits_x=8, **kwargs):
        super(DenseQ, self).__init__(units, **kwargs)

        self.range_w = 2**bits_w
        self.range_x = 2**bits_x
        self.kernel_q = None
        self.bias_q = None

    def build(self, input_shape):
        super(DenseQ, self).build(input_shape)
        self.kernel_q = K.round(self.range_w * self.kernel)
        self.bias_q = K.round(self.range_w * self.range_x * self.bias)

    def call(self, inputs, blind=True, bias=True):

        outputs = K.dot(inputs, self.kernel_q)

        if bias:
            outputs = K.bias_add(outputs, self.bias_q)

        return K.round(outputs / self.range_w)