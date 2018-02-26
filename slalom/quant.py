import tensorflow as tf
from keras.layers import Conv2D, Dense, initializers, regularizers, constraints, conv_utils
from keras.engine import InputSpec
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


class DepthwiseConv2D(Conv2D):
    """Depthwise separable 2D convolution.
    Depthwise Separable convolutions consists in performing
    just the first step in a depthwise spatial convolution
    (which acts on each input channel separately).
    The `depth_multiplier` argument controls how many
    output channels are generated per input channel in the depthwise step.
    # Arguments
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `'valid'` or `'same'` (case-insensitive).
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be 'channels_last'.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. 'linear' activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        depthwise_initializer: Initializer for the depthwise kernel matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        depthwise_regularizer: Regularizer function applied to
            the depthwise kernel matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its 'activation').
            (see [regularizer](../regularizers.md)).
        depthwise_constraint: Constraint function applied to
            the depthwise kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        4D tensor with shape:
        `[batch, channels, rows, cols]` if data_format='channels_first'
        or 4D tensor with shape:
        `[batch, rows, cols, channels]` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `[batch, filters, new_rows, new_cols]` if data_format='channels_first'
        or 4D tensor with shape:
        `[batch, new_rows, new_cols, filters]` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        outputs = K.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config


class DepthwiseConv2DQ(DepthwiseConv2D):
    def __init__(self, filters, kernel_size, bits_w=8, bits_x=8, **kwargs):
        super(DepthwiseConv2DQ, self).__init__(filters, kernel_size, **kwargs)
        self.quantize = QUANTIZE
        self.bits_w = bits_w
        self.bits_x = bits_x
        self.range_w = 2**bits_w
        self.range_x = 2**bits_x
        self.kernel_q = None
        self.bias_q = None

    def build(self, input_shape):
        super(DepthwiseConv2DQ, self).build(input_shape)
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

        base_config = super(DepthwiseConv2DQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))