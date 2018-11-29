import tensorflow as tf
from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, ZeroPadding2D, Flatten, GlobalAveragePooling2D, \
    Reshape, Dropout, AveragePooling2D, Lambda
from keras.layers import initializers
from keras.layers.normalization import BatchNormalization
from keras.utils import conv_utils
from keras.activations import relu, softmax, linear
from keras.models import Sequential
from keras.engine import InputLayer
from keras.engine.topology import Layer
from keras.applications.mobilenet import DepthwiseConv2D, relu6
import keras.backend as K
from python.slalom.resnet import ResNetBlock
from python.slalom.utils import get_all_layers 
import numpy as np
import itertools


# special Kernel to compute x + q * y (mod p)
fmod_module = tf.load_op_library('./App/cuda_fmod.so')

P = 2**23 + 2**21 + 7
INV_P = 1.0 / P
MID = P // 2
assert(P + MID < 2**24)
q = float(round(np.sqrt(MID))) + 1
inv_q = 1.0 / q


def remainder(x, p):
    return fmod_module.fmod(x, p=p)


def log2(x):
    num = tf.log(x)
    denom = tf.log(tf.constant(2, dtype=num.dtype))
    return num / denom


class Zeros64(initializers.Initializer):
    """Initializer that generates tensors initialized to 0.
    """

    def __call__(self, shape, dtype=None):
        return K.constant(0, shape=shape, dtype=tf.float64)


# quantized layer for any non-linear computation
class ActivationQ(Layer):

    def __init__(self, activation, bits_w, bits_x, maxpool_params=None, log=False, quantize=True,
                 slalom=False, slalom_integrity=False, slalom_privacy=False, 
                 sgxutils=None, queue=None, **kwargs):
        super(ActivationQ, self).__init__(**kwargs)
        self.bits_w = bits_w 
        self.bits_x = bits_x
        self.range_w = 2**bits_w
        self.range_x = 2**bits_x
        self.log = log
        self.quantize = quantize
        self.slalom = slalom
        self.slalom_integrity = slalom_integrity
        self.slalom_privacy = slalom_privacy
        self.sgxutils = sgxutils
        self.queue = queue
        self.activation = activation
        assert activation in ["relu", "relu6", "softmax", "avgpoolrelu", "avgpoolrelu6"]

        self.maxpool_params = maxpool_params
        if self.maxpool_params:
            assert self.quantize
            assert self.activation == "relu"

    def activation_name(self):
        if self.activation == "avgpoolrelu6":
            return "relu6"
        if self.activation == "avgpoolrelu":
            return "relu"
        return self.activation

    def call(self, inputs):
        #inputs = tf.Print(inputs, [tf.reduce_sum(tf.abs(tf.cast(inputs, tf.float64)))], message="relu input: ")
        #inputs = tf.Print(inputs, [], message="in ActivationQ with input shape: {}".format(inputs.get_shape().as_list()))

        if self.slalom:
            blind = self.queue.dequeue() if self.queue is not None else []
            if self.maxpool_params is not None:
                outputs = self.sgxutils.maxpoolrelu_slalom(inputs, blind, self.maxpool_params)
            else:
                outputs = self.sgxutils.relu_slalom(inputs, blind, activation=self.activation)
            if self.log:
                outputs = tf.Print(outputs, [tf.reduce_min(outputs), tf.reduce_max(outputs)], message="slalom output: ")
            return outputs

        if self.quantize and not self.slalom:
            if self.maxpool_params is not None:
                mp = self.maxpool_params
                outputs = K.round(inputs / self.range_w)
                outputs = K.pool2d(K.relu(outputs), mp['pool_size'],
                                   strides=mp['strides'], padding=mp['padding'], pool_mode='max')
                return outputs

            if self.activation in ["relu", "relu6", "avgpoolrelu", "avgpoolrelu6"]:
                if self.activation.endswith("relu6"):
                    act = K.relu(inputs, max_value=6 * self.range_x * self.range_w)
                else:
                    act = K.relu(inputs)

                if self.activation.startswith("avgpool"):
                    ch = inputs.get_shape()[3]
                    act = tf.reshape(K.mean(act, [1, 2]), (None, 1, 1, ch))

                outputs = K.round(act / self.range_w)

                if self.log:
                    outputs = tf.Print(outputs, [log2(tf.reduce_max(tf.abs(outputs))), tf.reduce_mean(tf.abs(outputs)),
                                         tf.greater_equal(log2(tf.reduce_max(tf.abs(outputs))), np.log2(MID))],
                                   message="Activation log: ")

                #outputs = tf.Print(outputs, [tf.reduce_sum(tf.abs(tf.cast(outputs, tf.float64)))], message="relu output: ")
                return outputs

            return inputs

        if self.activation in ["relu", "relu6", "avgpoolrelu", "avgpoolrelu6"]:
            if self.activation.endswith("relu6"):
                act = K.relu(inputs, max_value=6)
            else:
                act = K.relu(inputs)

            if self.activation.startswith("avgpool"):
                ch = inputs.get_shape().as_list()[3]
                act = tf.reshape(K.mean(act, [1, 2]), [-1, 1, 1, ch])
        else:
            act = inputs

        return act

    def get_config(self):
        config = {
            'bits_w': self.bits_w,
            'bits_x': self.bits_x,
            'quantize': self.quantize,
            'log': self.log,
            'slalom': self.slalom,
            'slalom_integrity': self.slalom_integrity,
            'slalom_privacy': self.slalom_privacy,
            'sgxutils': self.sgxutils,
            'queue': self.queue,
            'activation': self.activation,
            'maxpool_params': self.maxpool_params,
        }
        base_config = super(ActivationQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if self.activation.startswith("avgpool"):
            out_shape = (input_shape[0], 1, 1, input_shape[3])
            return out_shape

        if self.maxpool_params is not None:
            mp = self.maxpool_params
            rows = conv_utils.conv_output_length(input_shape[1], mp['pool_size'][0], mp['padding'], mp['strides'][0])
            cols = conv_utils.conv_output_length(input_shape[2], mp['pool_size'][1], mp['padding'], mp['strides'][1])
            return (input_shape[0], rows, cols, input_shape[3])

        return input_shape


# fuse batchnorm layers
def fuse_bn(layers):

    for (i, layer) in enumerate(layers):
        if isinstance(layer, BatchNormalization):
            input = layer.get_input_at(0)
            prev_layer = [l for l in layers if l.get_output_at(0) == input]
            assert len(prev_layer) == 1
            #conv = layers[i-1]
            conv = prev_layer[0]

            assert isinstance(conv, Conv2D) or isinstance(conv, DepthwiseConv2D)
            assert layer.axis == 3 or layer.axis == -1

            mean = layer.moving_mean
            var = layer.moving_variance
            beta = layer.beta if layer.beta is not None else 0.0
            gamma = layer.gamma if layer.gamma is not None else 1.0

            w = conv.get_weights()[0]
            b = 0

            # conv layer had no bias
            if not conv.use_bias:
                if isinstance(conv, DepthwiseConv2D):
                    bias_shape = (conv.depthwise_kernel.get_shape().as_list()[2],)
                else:
                    bias_shape = (conv.filters,)

                conv.add_weight(shape=bias_shape,
                                initializer=initializers.get('zeros'),
                                name='bias',
                                regularizer=None,
                                constraint=None)

                conv.use_bias = True
            else:
                b = conv.get_weights()[1]

            if isinstance(conv, DepthwiseConv2D):
                w = np.transpose(w, (0, 1, 3, 2))

            new_w = K.get_session().run(w * gamma / (K.sqrt(var + layer.epsilon)))
            new_b = K.get_session().run((b-mean) * gamma / (K.sqrt(var + layer.epsilon)) + beta)

            if isinstance(conv, DepthwiseConv2D):
                new_w = np.transpose(new_w, (0, 1, 3, 2))

            conv.set_weights((new_w, new_b)) 


# transform a model by quantizing and fusing layers
def transform(model, bits_w, bits_x, log=False, quantize=True, verif_preproc=False,
              slalom=False, slalom_integrity=False, slalom_privacy=False, sgxutils=None, queues=None):

    if slalom:
        assert(quantize)

    old_ops = K.get_session().graph.get_operations()

    #all_layers = [[l] if not isinstance(l, ResNetBlock) else l.get_layers() for l in model.layers]
    #all_layers = list(itertools.chain.from_iterable(all_layers))
    all_layers = get_all_layers(model)
    fuse_bn(all_layers)

    queue_ctr = 0
    layers = model.layers
    layer_map = {}
    flattened = False

    def transform_layer(layer, next_layer, queue_ctr, flattened):
        print("transform {} (next = {})".format(layer, next_layer))
        new_layers = []
        skip_next = False

        if isinstance(layer, InputLayer):
            new_layers.append(InputLayer.from_config(layer.get_config()))

        elif isinstance(layer, Conv2D) and not isinstance(layer, DepthwiseConv2D):
            conf = layer.get_config()

            act = conf['activation']

            # if the next layer is a pooling layer, create a fused activation
            maxpool_params = None
            if slalom and isinstance(next_layer, MaxPooling2D):
                mp = next_layer
                assert (layer.activation == relu)
                maxpool_params = mp.get_config()
                skip_next = True

            act_layer = None
            if act != "linear":
                conf['activation'] = "linear"

                if slalom and isinstance(next_layer, GlobalAveragePooling2D):
                    assert layer.activation in [relu, relu6]
                    act = "avgpool" + act
                    skip_next = True

                act_layer = ActivationQ(act, bits_w, bits_x, maxpool_params=maxpool_params, log=log,
                                        quantize=quantize, slalom=slalom, slalom_integrity=slalom_integrity,
                                        slalom_privacy=slalom_privacy, sgxutils=sgxutils,
                                        queue=None if queues is None else queues[queue_ctr])
                queue_ctr += 1

            conf['bits_w'] = bits_w
            conf['bits_x'] = bits_x
            conf['log'] = log
            conf['quantize'] = quantize
            conf['slalom'] = slalom
            conf['slalom_integrity'] = slalom_integrity
            conf['slalom_privacy'] = slalom_privacy
            conf['sgxutils'] = sgxutils

            new_layer = Conv2DQ.from_config(conf)
            new_layers.append(new_layer)
            layer_map[new_layer] = layer

            if act_layer is not None:
                new_layers.append(act_layer)

        elif isinstance(layer, DepthwiseConv2D):
            conf = layer.get_config()

            assert conf['activation'] == "linear"

            conf['bits_w'] = bits_w
            conf['bits_x'] = bits_x
            conf['log'] = log
            conf['quantize'] = quantize
            conf['slalom'] = slalom
            conf['slalom_integrity'] = slalom_integrity
            conf['slalom_privacy'] = slalom_privacy
            conf['sgxutils'] = sgxutils

            new_layer = DepthwiseConv2DQ.from_config(conf)
            new_layers.append(new_layer)
            layer_map[new_layer] = layer

        elif isinstance(layer, Dense):
            conf = layer.get_config()

            act = conf['activation']

            act_layer = None
            if act != "linear":
                conf['activation'] = "linear"
                act_layer = ActivationQ(act, bits_w, bits_x, log=log,
                                        quantize=quantize, slalom=slalom, slalom_integrity=slalom_integrity,
                                        slalom_privacy=slalom_privacy, sgxutils=sgxutils,
                                        queue=None if queues is None else queues[queue_ctr])
                queue_ctr += 1

            conf['bits_w'] = bits_w
            conf['bits_x'] = bits_x
            conf['log'] = log
            conf['quantize'] = quantize
            conf['slalom'] = slalom
            conf['slalom_integrity'] = slalom_integrity
            conf['slalom_privacy'] = slalom_privacy
            conf['sgxutils'] = sgxutils

            # replace the dense layer by a pointwise convolution
            if verif_preproc:
                del conf['units']
                conf['filters'] = layer.units
                conf['kernel_size'] = 1
                if not flattened:
                    h_in = int(layer.input_spec.axes[-1])
                    new_layers.append(Reshape((1, 1, h_in)))
                    flattened = True
                new_layer = Conv2DQ.from_config(conf)
                new_layers.append(new_layer)
                layer_map[new_layer] = layer

            else:
                new_layer = DenseQ.from_config(conf)
                new_layers.append(new_layer)
                layer_map[new_layer] = layer

            if act_layer is not None:
                new_layers.append(act_layer)

        elif isinstance(layer, BatchNormalization):
            pass

        elif isinstance(layer, MaxPooling2D):
            assert (not slalom or not slalom_privacy)
            new_layers.append(MaxPooling2D.from_config(layer.get_config()))

        elif isinstance(layer, AveragePooling2D):
            assert (not slalom or not slalom_privacy)
            new_layers.append(AveragePooling2D.from_config(layer.get_config()))
            new_layers.append(Lambda(lambda x: K.round(x)))

        elif isinstance(layer, Activation):
            assert layer.activation in [relu6, relu, softmax]

            queue = None if queues is None else queues[queue_ctr]
            queue_ctr += 1

            act_func = "relu6" if layer.activation == relu6 else "relu" if layer.activation == relu else "softmax"
            if slalom and isinstance(next_layer, GlobalAveragePooling2D):
                #assert layer.activation == relu6
                act_func = "avgpoolrelu6"
                skip_next = True

            maxpool_params = None
            if slalom and (isinstance(next_layer, MaxPooling2D) or isinstance(next_layer, AveragePooling2D)):
                mp = next_layer
                assert (layer.activation == relu)
                maxpool_params = mp.get_config()
                skip_next = True

            new_layers.append(ActivationQ(act_func, bits_w, bits_x, log=log,
                                      maxpool_params=maxpool_params,
                                      quantize=quantize, slalom=slalom,
                                      slalom_integrity=slalom_integrity,
                                      slalom_privacy=slalom_privacy,
                                      sgxutils=sgxutils, queue=queue))

        elif isinstance(layer, ZeroPadding2D):
            if quantize:
                # merge with next layer
                conv = next_layer 
                assert isinstance(conv, Conv2D) or isinstance(conv, DepthwiseConv2D)
                assert conv.padding == 'valid'
                conv.padding = 'same'
            else:
                new_layers.append(ZeroPadding2D.from_config(layer.get_config()))

        elif isinstance(layer, Flatten):
            if not verif_preproc:
                new_layers.append(Flatten.from_config(layer.get_config()))

        elif isinstance(layer, GlobalAveragePooling2D):
            assert not slalom
            conf = layer.get_config()
            conf['bits_w'] = bits_w
            conf['bits_x'] = bits_x
            conf['log'] = log
            conf['quantize'] = quantize
            new_layers.append(GlobalAveragePooling2DQ.from_config(conf))

        elif isinstance(layer, Reshape):
            new_layers.append(Reshape.from_config(layer.get_config()))

        elif isinstance(layer, Dropout):
            pass

        elif isinstance(layer, ResNetBlock):
            #assert not slalom

            path1 = []
            path2 = []
            for l in layer.path1:
                lq, queue_ctr, _, _ = transform_layer(l, None, queue_ctr, flattened)
                path1.extend(lq)

            for l in layer.path2:
                lq, queue_ctr, _, _ = transform_layer(l, None, queue_ctr, flattened)
                path2.extend(lq)

            [actq], queue_ctr, flattened, skip_next = transform_layer(layer.merge_act, next_layer, queue_ctr, flattened)
            new_layer = ResNetBlock(layer.kernel_size, layer.filters, layer.stage, layer.block, layer.identity,
                                    layer.strides, path1=path1, path2=path2, merge_act=actq, 
                                    quantize=quantize, bits_w=bits_w, bits_x=bits_x,
                                    slalom=slalom, slalom_integrity=slalom_integrity, slalom_privacy=slalom_privacy)

            new_layers.append(new_layer)
        else:
            raise AttributeError("Don't know how to handle layer {}".format(layer))

        return new_layers, queue_ctr, flattened, skip_next

    new_model = Sequential()
    skip_next = False
    while layers:
        layer = layers.pop(0)
        next_layer = layers[0] if len(layers) else None

        if not skip_next:
            new_layers, queue_ctr, flattened, skip_next = transform_layer(layer, next_layer, queue_ctr, flattened)
            for new_layer in new_layers:
                new_model.add(new_layer)
        else:
            skip_next = False

    print(new_model.summary())

    # copy over (and potentially quantize) all the weights
    new_layers = get_all_layers(new_model)

    for layer in new_layers:
        if layer in layer_map:
            src_layer = layer_map[layer]

            weights = src_layer.get_weights()
            kernel = weights[0]
            bias = weights[1]

            if quantize:
                range_w = 2**bits_w
                range_x = 2**bits_x
                kernel_q = np.round(range_w * kernel)
                bias_q = np.round(range_w * range_x * bias)
                if slalom_privacy:

                    if isinstance(layer, DepthwiseConv2DQ):
                        bias_q = bias_q.astype(np.float64)
                        kernel_q = kernel_q.astype(np.float64)

                layer._trainable_weights = layer._trainable_weights[2:]

                if isinstance(src_layer, Dense) and verif_preproc:
                    kernel_q = np.reshape(kernel_q, (1, 1, kernel_q.shape[0], kernel_q.shape[1]))

                layer.set_weights((kernel_q, bias_q))
            else:
                layer._trainable_weights = layer._trainable_weights[2:]
                layer.set_weights((kernel, bias))

    # find all the TensorFlow ops that correspond to inputs/outputs of linear operators
    new_ops = [op for op in K.get_session().graph.get_operations() if op not in old_ops]
    linear_ops_in = [tf.reshape(op.inputs[0], [-1]) for op in new_ops if op.type in ['Conv2D', 'MatMul', 'DepthwiseConv2dNative']]
    linear_ops_out = [tf.reshape(op.outputs[0], [-1]) for op in new_ops if op.type in ['BiasAdd']]

    return new_model, linear_ops_in, linear_ops_out


# build operations for computing unblinding factors
def build_blinding_ops(model, queues, batch_size):
    linear_layers = get_all_linear_layers(model)
    print("preparing blinding factors for {} layers...".format(len(linear_layers)))

    assert(len(queues) == len(linear_layers))

    in_placeholders = [tf.placeholder(tf.float32, shape=(batch_size,) + layer.input_shape[1:]) for layer in linear_layers]
    zs = [layer.call(ph, early_return='prod') for (layer, ph) in zip(linear_layers, in_placeholders)]
    temps = [layer.call(ph, early_return='bias') for (layer, ph) in zip(linear_layers, in_placeholders)]
    out_funcs = [None] * len(linear_layers)

    out_placeholders = [tf.placeholder(tf.float32, shape=z.get_shape().as_list()) for (layer, z) in zip(linear_layers, zs)]
    queue_ops = [q.enqueue(ph) for (q, ph) in zip(queues, out_placeholders)]

    return in_placeholders, zs, out_placeholders, queue_ops, temps, out_funcs


# prepare unblinding factors. For convenience this is currently done outside the enclave
def prepare_blinding_factors(sess, model, sgxutils, in_placeholders, zs, out_placeholders, queue_ops,
                             batch_size, num_batches=1, inputs=None, temps=None, out_funcs=None):

    linear_layers = get_all_linear_layers(model)

    for i in range(num_batches):
        if inputs is not None:
            curr_input = inputs[i:i+1]

        print("batch {}/{}".format(i+1, num_batches))
        for j, layer in enumerate(linear_layers):
            print()
            print("\tlayer {}/{} ({}, {})".format(j+1, len(linear_layers), layer, layer.activation))
            shape = (batch_size,) + layer.input_shape[1:]

            # get the blinding factor from the enclave
            r = np.zeros(shape=shape, dtype=np.float32)
            sgxutils.slalom_get_r(r)
            #r = np.random.randint(low=-MID, high=MID+1, size=shape).astype(np.float32)

            print("r: {}".format((r.shape, np.min(r), np.max(r), np.sum(np.abs(r.astype(np.float64))))))
            #assert((np.round(r) == r).all())

            # compute the unblinding factor
            z = sess.run(zs[j], feed_dict={in_placeholders[j]: r})
            print("z: {}".format((z.shape, np.min(z), np.max(z), np.sum(np.abs(z.astype(np.float64))))))

            # debug with real data
            if inputs is not None:
                inp = curr_input
                xr = inp.astype(np.float64) + r.astype(np.float64)
                xr[xr >= MID] -= P
                xr[xr < -MID] += P
                xr = xr.astype(np.float32)

                print("blinded input: {}".format((np.min(xr), np.max(xr), np.sum(np.abs(xr.astype(np.float64))))))

                # Conv(x, w) + b
                real = sess.run(temps[j], feed_dict={in_placeholders[j]: inp})

                # Conv(x, w) + Conv(r, w) + b
                blind = sess.run(temps[j], feed_dict={in_placeholders[j]: xr})

                print("blinded output: {}".format((np.min(blind), np.max(blind), np.sum(np.abs(blind.astype(np.float64))))))

                # Conv(x, w) + b
                unblind = blind.astype(np.float64) - z.astype(np.float64)
                unblind[unblind >= MID] -= P
                unblind[unblind < -MID] += P
                unblind = unblind.astype(np.float32)

                if not (real == unblind).all():
                    print("real output: {}".format((np.min(real), np.max(real), np.sum(np.abs(real.astype(np.float64))))))
                    print("================FAILED ON LAYER {}================".format(j))
                    assert(0)

                if out_funcs[j] is None:
                    ph = tf.placeholder(dtype=tf.float32, shape=real.shape)
                    if hasattr(layer, 'maxpool_params') and layer.maxpool_params is not None:
                        f0 = tf.nn.max_pool(tf.nn.relu(ph), (1, 2, 2, 1), (1, 2, 2, 1), "SAME")
                    elif layer.activation.__name__ == 'relu':
                        f0 = tf.nn.relu(ph)
                    else:
                        assert(layer.activation.__name__ in ['linear', 'softmax'])
                        # last layer has no activation
                        f0 = ph

                    if j < len(linear_layers) - 1:
                        shape = (batch_size,) + linear_layers[j+1].input_shape[1:]
                    else:
                        shape = real.shape

                    f = K.round(tf.reshape(f0, shape) / 2**8)
                    out_funcs[j] = (f, ph)

                (f, ph) = out_funcs[j]
                curr_input = sess.run(f, feed_dict={ph: real})
                print("non-linear output: {}".format((np.min(curr_input), np.max(curr_input), np.sum(np.abs(curr_input.astype(np.float64))))))

            z_enc = np.zeros(shape=z.shape, dtype=np.float32)
            sgxutils.slalom_set_z(z, z_enc)
            sess.run(queue_ops[j], feed_dict={out_placeholders[j]: z_enc})

    print("blinding factors done")
    print()


# quantized convolution
class Conv2DQ(Conv2D):
    def __init__(self, filters, kernel_size, bits_w=8, bits_x=8, quantize=True, log=False,
                 slalom=False, slalom_integrity=False, slalom_privacy=False, sgxutils=None, queue=None, **kwargs):
        super(Conv2DQ, self).__init__(filters, kernel_size, **kwargs)
        self.quantize = quantize
        self.log = log
        self.slalom = slalom
        self.slalom_integrity = slalom_integrity
        self.slalom_privacy = slalom_privacy
        self.bits_w = bits_w
        self.bits_x = bits_x
        self.range_w = 2**bits_w
        self.range_x = 2**bits_x
        self.sgxutils = sgxutils
        self.queue = queue
        self.is_pointwise = (kernel_size == 1) or (kernel_size == (1, 1))

        assert self.activation == linear

    def build(self, input_shape):
        super(Conv2DQ, self).build(input_shape)

        kernel_type = tf.float32
        kernel_init = initializers.get('zeros')
        self.kernel_q = self.add_weight(shape=self.kernel.get_shape().as_list(),
                                        dtype=kernel_type,
                                        initializer=kernel_init,
                                        name='kernel_q')

        bias_type = tf.float32
        bias_init = initializers.get('zeros')
        self.bias_q = self.add_weight(shape=self.bias.get_shape().as_list(),
                                      dtype=bias_type,
                                      initializer=bias_init,
                                      name='bias_q')

    def compute_output_shape(self, input_shape):
        return super(Conv2DQ, self).compute_output_shape(input_shape)

    def call(self, inputs, early_return=None):
        #inputs = tf.Print(inputs, [tf.reduce_sum(tf.abs(tf.cast(inputs, tf.float64)))], message="conv input: ")

        if early_return is not None:
            assert early_return in ['prod', 'bias']

        if not self.slalom_privacy:
            outputs = tf.nn.conv2d(
                input=inputs,
                filter=self.kernel_q,
                strides=(1,) + self.strides + (1,),
                padding=self.padding.upper(),
                data_format='NHWC')
            if self.use_bias and not early_return == 'prod':
                outputs = K.bias_add(outputs, self.bias_q)
        else:
            if not self.is_pointwise or self.strides != (1, 1):
                # split inputs into two halves to avoid overflowing the single-precision floating point representation
                # inputs = inputs_low + q * inputs_high

                inputs_low = remainder(inputs, q)
                inputs_high = tf.round((inputs - inputs_low) / q)

                outputs_low = tf.nn.conv2d(
                    input=inputs_low,
                    filter=self.kernel_q,
                    strides=(1,) + self.strides + (1,),
                    padding=self.padding.upper(),
                    data_format='NHWC')

                outputs_high = tf.nn.conv2d(
                    input=inputs_high,
                    filter=self.kernel_q,
                    strides=(1,) + self.strides + (1,),
                    padding=self.padding.upper(),
                    data_format='NHWC')

                # reconstruct result and take mod p (lands in [-p, p]. this is fine since we're only taking this
                # modulo to send less stuff to the enclave)
                if self.use_bias and not early_return == 'prod':
                    outputs_low = K.bias_add(outputs_low, self.bias_q, data_format=self.data_format)

                outputs = fmod_module.mod_cast(outputs_low, outputs_high, q=q, p=P)

            else:
                # same but for pointwise convolution
                h = inputs.get_shape().as_list()[1]
                w = inputs.get_shape().as_list()[2]
                ch_in = inputs.get_shape().as_list()[3]
                ch_out = self.kernel_q.get_shape().as_list()[-1]

                inputs = tf.reshape(inputs, (-1, ch_in))
                inputs_low = remainder(inputs, q)
                inputs_high = tf.round((inputs - inputs_low) / q)
                outputs_low = K.dot(inputs_low, tf.reshape(self.kernel_q, (ch_in, ch_out)))
                outputs_high = K.dot(inputs_high, tf.reshape(self.kernel_q, (ch_in, ch_out)))
                if self.use_bias and not early_return == 'prod':
                    outputs_low = K.bias_add(outputs_low, self.bias_q, data_format=self.data_format)

                outputs = fmod_module.mod_cast(outputs_low, outputs_high, q=q, p=P)
                outputs = tf.reshape(outputs, (-1, h, w, ch_out))

        if self.log:
            outputs = tf.Print(outputs, [log2(tf.reduce_max(tf.abs(outputs))), tf.reduce_mean(tf.abs(outputs)),
                                         tf.greater_equal(log2(tf.reduce_max(tf.abs(outputs))), np.log2(MID))],
                               message="Conv2D log: ")

        if early_return == 'prod' or early_return == 'bias':
            # Conv(Z, w)
            return tf.cast(remainder(outputs, P), tf.float32)

        #outputs = tf.Print(outputs, [tf.reduce_sum(tf.abs(tf.cast(outputs, tf.float64)))], message="conv output: ")
        return outputs

    def get_config(self):
        config = {
            'bits_w': self.bits_w,
            'bits_x': self.bits_x,
            'quantize': self.quantize,
            'log': self.log,
            'slalom': self.slalom,
            'slalom_integrity': self.slalom_integrity,
            'slalom_privacy': self.slalom_privacy,
            'sgxutils': self.sgxutils,
            'queue': self.queue
        }
        
        base_config = super(Conv2DQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# quantized dense layer
class DenseQ(Dense):

    def __init__(self, units, bits_w=8, bits_x=8, quantize=True, log=False,
                 slalom=False, slalom_integrity=False, slalom_privacy=False, sgxutils=None, queue=None, **kwargs):
        super(DenseQ, self).__init__(units, **kwargs)

        self.quantize = quantize
        self.log = log
        self.slalom = slalom
        self.slalom_integrity = slalom_integrity
        self.slalom_privacy = slalom_privacy

        self.bits_w = bits_w
        self.bits_x = bits_x
        self.range_w = 2**bits_w
        self.range_x = 2**bits_x
        self.kernel_q = None
        self.bias_q = None

        self.sgxutils = sgxutils
        self.queue = queue

        assert self.activation == linear

    def build(self, input_shape):
        super(DenseQ, self).build(input_shape)

        kernel_type = tf.float32
        kernel_init = initializers.get('zeros')
        self.kernel_q = self.add_weight(shape=self.kernel.get_shape().as_list(),
                                        dtype=kernel_type,
                                        initializer=kernel_init,
                                        name='kernel_q')

        bias_type = tf.float32
        bias_init = initializers.get('zeros')
        self.bias_q = self.add_weight(shape=self.bias.get_shape().as_list(),
                                      dtype=bias_type,
                                      initializer=bias_init,
                                      name='bias_q')

    def call(self, inputs, early_return=None):
        #inputs = tf.Print(inputs, [tf.reduce_sum(tf.abs(tf.cast(inputs, tf.float64)))], message="dense input: ")

        if early_return is not None:
            assert early_return in ['prod', 'bias']

        if self.slalom_privacy:
            # split inputs into two halves to avoid overflowing the single-precision floating point representation
            # inputs = inputs_low + q * inputs_high

            inputs_low = remainder(inputs, q)
            inputs_high = tf.round((inputs - inputs_low) / q)
            outputs_low = K.dot(inputs_low, self.kernel_q)
            outputs_high = K.dot(inputs_high, self.kernel_q)
            if self.use_bias and not early_return == 'prod':
                outputs_low = K.bias_add(outputs_low, self.bias_q)
            outputs = tf.cast(outputs_low, tf.float64) + q * tf.cast(outputs_high, tf.float64)
        else:
            outputs = K.dot(inputs, self.kernel_q)
            if self.use_bias and not early_return == 'prod':
                outputs = K.bias_add(outputs, self.bias_q)

        if self.log:
            outputs = tf.Print(outputs, [log2(tf.reduce_max(tf.abs(outputs))), tf.reduce_mean(tf.abs(outputs)),
                                         tf.greater_equal(log2(tf.reduce_max(tf.abs(outputs))), np.log2(MID))],
                               message="dense log: ")

        if early_return == 'prod' or early_return == 'bias':
            # Z.dot(W)
            return tf.cast(remainder(outputs, P), tf.float32)

        if self.slalom_privacy:
            outputs = tf.cast(remainder(outputs, P), tf.float32)

        #outputs = tf.Print(outputs, [tf.reduce_sum(tf.abs(tf.cast(outputs, tf.float64)))], message="dense output: ")
        return outputs

    def get_config(self):
        config = {
            'bits_w': self.bits_w,
            'bits_x': self.bits_x,
            'quantize': self.quantize,
            'log': self.log,
            'slalom': self.slalom,
            'slalom_integrity': self.slalom_integrity,
            'slalom_privacy': self.slalom_privacy,
            'sgxutils': self.sgxutils,
            'queue': self.queue
        }

        base_config = super(DenseQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# quantized depthwise convolution
class DepthwiseConv2DQ(DepthwiseConv2D):
    def __init__(self, kernel_size, bits_w=8, bits_x=8, quantize=True, log=False,
                 slalom=False, slalom_integrity=False, slalom_privacy=False, sgxutils=None, queue=None, **kwargs):
        super(DepthwiseConv2DQ, self).__init__(kernel_size, **kwargs)

        self.quantize = quantize
        self.log = log 
        self.slalom = slalom
        self.slalom_integrity = slalom_integrity
        self.slalom_privacy = slalom_privacy

        self.bits_w = bits_w
        self.bits_x = bits_x
        self.range_w = 2**bits_w
        self.range_x = 2**bits_x

        self.sgxutils = sgxutils
        self.queue = queue

        assert self.activation == linear

    def build(self, input_shape):
        super(DepthwiseConv2DQ, self).build(input_shape)

        weight_type = tf.float64 if self.slalom_privacy else tf.float32
        init = Zeros64() if self.slalom_privacy else initializers.get('zeros')
        self.kernel_q = self.add_weight(shape=self.depthwise_kernel.get_shape().as_list(),
                                        dtype=weight_type,
                                        initializer=init,
                                        name='kernel_q')

        self.bias_q = self.add_weight(shape=self.bias.get_shape().as_list(),
                                      dtype=weight_type,
                                      initializer=init,
                                      name='bias_q')

    def call(self, inputs, early_return=None):

        if early_return is not None:
            assert early_return in ['prod', 'bias']

        if self.slalom_privacy:
            inputs = K.cast(inputs, tf.float64)

        outputs = K.depthwise_conv2d(
            inputs,
            self.kernel_q,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.log:
            outputs = tf.Print(outputs, [log2(tf.reduce_max(tf.abs(outputs))), tf.reduce_mean(tf.abs(outputs)),
                                         tf.greater_equal(log2(tf.reduce_max(tf.abs(outputs))), np.log2(MID))],
                               message="DepthConv log: ")

        if early_return == 'prod':
            # X .* W
            return tf.cast(remainder(outputs, P), tf.float32)

        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias_q, data_format=self.data_format)

        if self.slalom_privacy:
            outputs = tf.cast(remainder(outputs, P), tf.float32)

        if early_return == 'bias':
            # X .* W + b
            return outputs

        return outputs

    def get_config(self):
        config = {
            'bits_w': self.bits_w,
            'bits_x': self.bits_x,
            'quantize': self.quantize,
            'log': self.log,
            'slalom': self.slalom,
            'slalom_integrity': self.slalom_integrity,
            'slalom_privacy': self.slalom_privacy,
            'sgxutils': self.sgxutils,
            'queue': self.queue
        } 

        base_config = super(DepthwiseConv2DQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# quantized global average pooling layer
class GlobalAveragePooling2DQ(GlobalAveragePooling2D):

    def __init__(self, bits_w, bits_x, quantize=True, log=False, **kwargs):
        super(GlobalAveragePooling2DQ, self).__init__(**kwargs)
        self.quantize = quantize
        self.log = log
        self.bits_w = bits_w
        self.bits_x = bits_x
        self.range_w = 2**bits_w
        self.range_x = 2**bits_x

    def call(self, inputs):
        ch = inputs.get_shape().as_list()[3]
        res = tf.reshape(K.mean(inputs, axis=[1, 2]), [-1, 1, 1, ch])
        if self.quantize:
            return K.round(res)

        if self.log:
            res = tf.Print(res, [log2(tf.reduce_max(tf.abs(res))), tf.reduce_mean(tf.abs(res)),
                                         tf.greater_equal(log2(tf.reduce_max(tf.abs(res))), np.log2(MID))],
                               message="AvgPool log: ")

        return res

    def compute_output_shape(self, input_shape):
        out_shape = (input_shape[0], 1, 1, input_shape[3])
        return out_shape

    def get_config(self):
        config = {
            'bits_w': self.bits_w,
            'bits_x': self.bits_x,
            'quantize': self.quantize,
            'log': self.log
        }
        base_config = super(GlobalAveragePooling2DQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_all_linear_layers(model):
    layers = []
    all_layers = get_all_layers(model)
    for idx, layer in enumerate(all_layers):
        # check if linear layer
        if hasattr(layer, 'kernel') or hasattr(layer, 'depthwise_kernel'):
            if layer.activation.__name__ != "linear":
                layers.append(layer)
            else:
                next_layer = all_layers[idx+1] if idx+1 < len(all_layers) else None

                if isinstance(next_layer, BatchNormalization):
                    next_layer = all_layers[idx+2] if idx+2 < len(all_layers) else None

                if isinstance(next_layer, Activation) or isinstance(next_layer, ActivationQ):
                    layers.append(layer)

    return layers
 
