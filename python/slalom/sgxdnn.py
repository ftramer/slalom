from ctypes import *
from ctypes import POINTER
import json

import numpy as np

import keras
from keras import activations
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, DepthwiseConv2D, GlobalAveragePooling2D, Dropout, \
    Reshape, ZeroPadding2D, AveragePooling2D, Lambda
from python.slalom.quant_layers import Conv2DQ, DenseQ, DepthwiseConv2DQ, ActivationQ
from python.slalom.resnet import ResNetBlock
import tensorflow as tf


SGXDNNLIB = "App/enclave_bridge.so"
DNNLIB = "lib/sgxdnn.so"

SGX_SLALOM_LIB = "lib/slalom_ops_sgx.so"
SLALOM_LIB = "lib/slalom_ops.so"


# interface with the C++ SGXDNN library
class SGXDNNUtils(object):

    def __init__(self, use_sgx=False, num_enclaves=1):
        self.use_sgx = use_sgx
        if use_sgx:
            self.lib = cdll.LoadLibrary(SGXDNNLIB)
            self.lib.initialize_enclave.restype = c_ulong
            self.eid = []
            for i in range(num_enclaves): 
                self.eid.append(self.lib.initialize_enclave())

            self.slalom_lib = tf.load_op_library(SGX_SLALOM_LIB)
        else:
            self.lib = cdll.LoadLibrary(DNNLIB)
            self.eid = None
            self.slalom_lib = tf.load_op_library(SLALOM_LIB)

    def destroy(self):
        if self.use_sgx and self.eid is not None:
            self.lib.destroy_enclave.argtypes = [c_ulong]
            for eid in self.eid:
                self.lib.destroy_enclave(eid)
            self.eid = None

    def benchmark(self, num_threads):
        if self.use_sgx:
            self.lib.sgxdnn_benchmarks.argtypes = [c_ulong, c_int]
            self.lib.sgxdnn_benchmarks(self.eid[0], num_threads)
        else:
            self.lib.sgxdnn_benchmarks(num_threads)

    def load_model(self, model_json, weights, dtype=np.float32, verify=False, verify_preproc=False):

        assert np.all([w.dtype == dtype for w in weights])

        assert dtype == np.float32
        print("loading model in float32")
        ptr_type = c_float
        if verify:
            load_method = self.lib.load_model_float_verify
        else:
            load_method = self.lib.load_model_float

        filter_ptrs = (POINTER(ptr_type) * len(weights))()
        for i in range(len(weights)):
            filter_ptrs[i] = np.ctypeslib.as_ctypes(weights[i])

        print("loading model...")
        if self.use_sgx:
            if verify:
                load_method.argtypes = [c_ulong, c_char_p, POINTER(POINTER(ptr_type)), c_bool]
                for eid in self.eid:
                    load_method(eid, json.dumps(model_json).encode('utf-8'), filter_ptrs, verify_preproc)
            else:
                load_method.argtypes = [c_ulong, c_char_p, POINTER(POINTER(ptr_type))]
                for eid in self.eid:
                    load_method(eid, json.dumps(model_json).encode('utf-8'), filter_ptrs)
        else:
            if verify:
                load_method.argtypes = [c_char_p, POINTER(POINTER(ptr_type)), c_bool]
                load_method(json.dumps(model_json).encode('utf-8'), filter_ptrs, verify_preproc)
            else:
                load_method.argtypes = [c_char_p, POINTER(POINTER(ptr_type))]
                load_method(json.dumps(model_json).encode('utf-8'), filter_ptrs)
        print("model loaded")

    def predict(self, x, num_classes=1000, eid_idx=0):
        dtype = np.float32
        x_typed = x.reshape(-1).astype(dtype)
        inp_ptr = np.ctypeslib.as_ctypes(x_typed)

        res = np.zeros((len(x), num_classes), dtype=dtype)
        res_ptr = np.ctypeslib.as_ctypes(res.reshape(-1))

        ptr_type = c_float
        predict_method = self.lib.predict_float

        if self.use_sgx:
            predict_method.argtypes = [c_ulong, POINTER(ptr_type), POINTER(ptr_type), c_int]
            predict_method(self.eid[eid_idx], inp_ptr, res_ptr, x.shape[0])
        else:
            predict_method.argtypes = [POINTER(ptr_type), POINTER(ptr_type), c_int]
            predict_method(inp_ptr, res_ptr, x.shape[0])

        return res

    def predict_and_verify(self, x, aux_data, num_classes=1000, dtype=np.float64, eid_idx=0):
        assert dtype == np.float32
        ptr_type = c_float
        predict_method = self.lib.predict_verify_float
        ptr_type_aux = c_float

        x_typed = x.reshape(-1).astype(dtype)
        inp_ptr = np.ctypeslib.as_ctypes(x_typed)

        aux_ptrs = (POINTER(ptr_type_aux) * len(aux_data))()
        for i in range(len(aux_data)):
            aux_ptrs[i] = np.ctypeslib.as_ctypes(aux_data[i])

        res = np.zeros((len(x), num_classes), dtype=dtype)
        res_ptr = np.ctypeslib.as_ctypes(res.reshape(-1))

        if self.use_sgx:
            predict_method.argtypes = [c_ulong, POINTER(ptr_type), POINTER(ptr_type),
                                       POINTER(POINTER(ptr_type_aux)), c_int]
            predict_method(self.eid[eid_idx], inp_ptr, res_ptr, aux_ptrs, x.shape[0])
        else:
            predict_method.argtypes = [POINTER(ptr_type), POINTER(ptr_type),
                                       POINTER(POINTER(ptr_type_aux)), c_int]
            predict_method(inp_ptr, res_ptr, aux_ptrs, x.shape[0])
        return res.astype(np.float32)

    def relu_slalom(self, inputs, blind, activation, eid_idx=0):
        if self.use_sgx:
            eid = self.eid[eid_idx]
            eid_low = eid % 2**32
            eid_high = eid // 2**32
            return self.slalom_lib.relu_slalom(inputs, blind, activation=activation, eid_low=eid_low, eid_high=eid_high)
        else:
            return self.slalom_lib.relu_slalom(inputs, blind, activation=activation)

    def maxpoolrelu_slalom(self, inputs, blind, params, eid_idx=0):
        ksize = (1, params['pool_size'][0], params['pool_size'][1], 1)
        strides = (1, params['strides'][0], params['strides'][1], 1)
        padding = params['padding'].upper()

        if self.use_sgx:
            eid = self.eid[eid_idx]
            eid_low = eid % 2**32
            eid_high = eid // 2**32
            return self.slalom_lib.relu_max_pool_slalom(inputs, blind, ksize, strides, padding, eid_low=eid_low, eid_high=eid_high)
        else:
            return self.slalom_lib.relu_max_pool_slalom(inputs, blind, ksize, strides, padding)

    def slalom_init(self, slalom_integrity, slalom_privacy, batch_size, eid_idx=0):
        if self.use_sgx:
            self.lib.slalom_init.argtypes = [c_ulong, c_bool, c_bool, c_int]
            self.lib.slalom_init(self.eid[eid_idx], slalom_integrity, slalom_privacy, batch_size)
        else:
            self.lib.slalom_init(slalom_integrity, slalom_privacy, batch_size)

    def slalom_get_r(self, r, eid_idx=0):
        r_flat = r.reshape(-1)
        inp_ptr = np.ctypeslib.as_ctypes(r_flat)

        if self.use_sgx:
            self.lib.slalom_get_r.argtypes = [c_ulong, POINTER(c_float), c_int]
            self.lib.slalom_get_r(self.eid[eid_idx], inp_ptr, r.size)
        else:
            self.lib.slalom_get_r.argtypes = [POINTER(c_float), c_int]
            self.lib.slalom_get_r(inp_ptr, r.size)

    def slalom_set_z(self, z, z_enc, eid_idx=0):
        z_flat = z.reshape(-1)
        inp_ptr = np.ctypeslib.as_ctypes(z_flat)

        z_enc_flat = z_enc.reshape(-1)
        out_ptr = np.ctypeslib.as_ctypes(z_enc_flat)

        if self.use_sgx:
            self.lib.slalom_set_z.argtypes = [c_ulong, POINTER(c_float), POINTER(c_float), c_int]
            self.lib.slalom_set_z(self.eid[eid_idx], inp_ptr, out_ptr, z.size)
        else:
            self.lib.slalom_set_z.argtypes = [POINTER(c_float), POINTER(c_float), c_int]
            self.lib.slalom_set_z(inp_ptr, out_ptr, z.size)
   
    def align_numpy(self, unaligned):
        sess = tf.get_default_session()
        aligned = sess.run(tf.ones(unaligned.shape, dtype=unaligned.dtype))
        np.copyto(aligned, unaligned)
        return aligned
 
    def slalom_blind_input(self, x, eid_idx=0):
        res = self.align_numpy(x)
        res_flat = res.reshape(-1)
        inp_ptr = np.ctypeslib.as_ctypes(res_flat)
        out_ptr = np.ctypeslib.as_ctypes(res_flat)

        if self.use_sgx:
            self.lib.slalom_blind_input.argtypes = [c_ulong, POINTER(c_float), POINTER(c_float), c_int]
            self.lib.slalom_blind_input(self.eid[eid_idx], inp_ptr, out_ptr, x.size)
        else:
            self.lib.slalom_blind_input.argtypes = [POINTER(c_float), POINTER(c_float), c_int]
            self.lib.slalom_blind_input(inp_ptr, out_ptr, x.size)
        
        return res


# convert a model to JSON format and potentially precompute integrity check vectors
def model_to_json(sess, model, verif_preproc=False, slalom_privacy=False, dtype=np.float32,
                  bits_w=0, bits_x=0):

    def get_activation_name(layer):
        if layer.activation is not None:
            return activations.serialize(layer.activation)
        else:
            return ''

    p = ((1 << 24) - 3) 
    r_max = (1 << 19)
    reps = 2

    def layer_to_json(layer):
        json = {}
        layer_weights = []

        if isinstance(layer, keras.layers.InputLayer):
            json = {'name': 'Input', 'shape': layer.batch_input_shape}

        elif isinstance(layer, Conv2D) and not isinstance(layer, DepthwiseConv2D):
            json = {'name': 'Conv2D',
                    'kernel_size': layer.kernel.get_shape().as_list(),
                    'strides': layer.strides,
                    'padding': layer.padding,
                    'activation': get_activation_name(layer),
                    'bits_w': layer.bits_w, 'bits_x': layer.bits_x}

            if isinstance(layer, Conv2DQ):
                kernel = sess.run(layer.kernel_q)
                bias = sess.run(layer.bias_q)

                print(layer, layer.input_shape, layer.output_shape, kernel.shape)

                if verif_preproc:
                    # precompute W*r or r_left * W * r_right
                    k_w, k_h, ch_in, ch_out = kernel.shape
                    h, w = layer.input_shape[1], layer.input_shape[2]
                    h_out, w_out = layer.output_shape[1], layer.output_shape[2]

                    np.random.seed(0)
                    if k_w == 1 and k_h == 1:
                        # pointwise conv
                        r_left = np.array([]).astype(np.float32)
                        r_right = np.ones(shape=(reps, ch_out)).astype(np.float32)
                        w_r = kernel.astype(np.float64).reshape((-1, ch_out)).dot(r_right.T.astype(np.float64))
                        w_r = w_r.T
                        w_r = np.fmod(w_r, p).astype(np.float32)
                        assert np.max(np.abs(w_r)) < 2 ** 52
                        b_r = np.fmod(np.dot(bias.astype(np.float64), r_right.T.astype(np.float64)), p).astype(
                            np.float32)
                    else:
                        r_left = np.random.randint(low=-r_max, high=r_max + 1, size=(reps, h_out * w_out)).astype(
                            np.float32)
                        r_right = np.random.randint(low=-r_max, high=r_max + 1, size=(reps, ch_out)).astype(np.float32)
                        w_r = np.zeros((reps, h * w, ch_in)).astype(np.float32)
                        b_r = np.zeros(reps).astype(np.float32)

                        X = np.zeros((1, h, w, ch_in)).astype(np.float64)
                        x_ph = tf.placeholder(tf.float64, shape=(None, h, w, ch_in))
                        w_ph = tf.placeholder(tf.float64, shape=(k_h, k_w, ch_in, 1))
                        y_ph = tf.placeholder(tf.float64, shape=(1, h_out, w_out, 1))
                        z = tf.nn.conv2d(x_ph, w_ph, (1,) + layer.strides + (1,), layer.padding.upper())
                        dz = tf.gradients(z, x_ph, grad_ys=y_ph)[0]

                        for i in range(reps):
                            curr_r_left = r_left[i].astype(np.float64)
                            curr_r_right = r_right[i].astype(np.float64)
                            #print("sum(curr_r_left) = {}".format(np.sum(curr_r_left)))
                            #print("sum(curr_r_right) = {}".format(np.sum(curr_r_right)))

                            w_right = kernel.astype(np.float64).reshape((-1, ch_out)).dot(curr_r_right)
                            #print("sum(w_right) = {}".format(np.sum(w_right)))
                            assert np.max(np.abs(w_right)) < 2 ** 52

                            w_r_i = sess.run(dz, feed_dict={x_ph: X, w_ph: w_right.reshape(k_w, k_h, ch_in, 1),
                                                            y_ph: curr_r_left.reshape((1, h_out, w_out, 1))})
                            #print("sum(w_r) = {}".format(np.sum(w_r_i.astype(np.float64))))
                            w_r[i] = np.fmod(w_r_i, p).astype(np.float32).reshape((h * w, ch_in))
                            assert np.max(np.abs(w_r[i])) < 2 ** 52
                            #print("sum(w_r) = {}".format(np.sum(w_r[i].astype(np.float64))))
                            b_r[i] = np.fmod(
                                np.sum(curr_r_left) * np.fmod(np.dot(bias.astype(np.float64), curr_r_right), p),
                                p).astype(np.float32)
                            #print("sum(b_r) = {}".format(np.sum(b_r[i].astype(np.float64))))

                    print("r_left: {}".format(r_left.astype(np.float64).sum()))
                    print("r_right: {}".format(r_right.astype(np.float64).sum()))
                    print("w_r: {}".format(w_r.astype(np.float64).sum()))
                    print("b_r: {}".format(b_r.astype(np.float64).sum()))
                    layer_weights.append(r_left.reshape(-1))
                    layer_weights.append(r_right.reshape(-1))
                    layer_weights.append(w_r.reshape(-1))
                    layer_weights.append(b_r.reshape(-1))
            else:
                kernel = layer.kernel.eval(sess)
                bias = layer.bias.eval(sess)
                print("sum(abs(conv_w)): {}".format(np.abs(kernel).sum()))

            if not verif_preproc:
                layer_weights.append(kernel.reshape(-1).astype(dtype))
                layer_weights.append(bias.reshape(-1).astype(dtype))

        elif isinstance(layer, MaxPooling2D):
            json = {'name': 'MaxPooling2D', 'pool_size': layer.pool_size,
                    'strides': layer.strides, 'padding': layer.padding}

        elif isinstance(layer, AveragePooling2D):
            json = {'name': 'AveragePooling2D', 'pool_size': layer.pool_size,
                    'strides': layer.strides, 'padding': layer.padding}

        elif isinstance(layer, Flatten):
            json = {'name': 'Flatten'}

        elif isinstance(layer, Dense):
            assert not (slalom_privacy and verif_preproc)
            json = {'name': 'Dense', 'kernel_size': layer.kernel.get_shape().as_list(),
                    'pointwise_conv': False, 'activation': get_activation_name(layer),
                    'bits_w': layer.bits_w, 'bits_x': layer.bits_x}

            if isinstance(layer, DenseQ):
                kernel = sess.run(layer.kernel_q).reshape(-1).astype(dtype)
                bias = sess.run(layer.bias_q).reshape(-1).astype(dtype)

            else:
                kernel = layer.kernel.eval(sess).reshape(-1).astype(dtype)
                bias = layer.bias.eval(sess).reshape(-1).astype(dtype)
            print("sum(abs(dense_w)): {}".format(np.abs(kernel).sum()))
            layer_weights.append(kernel)
            layer_weights.append(bias)

        elif isinstance(layer, DepthwiseConv2D):
            json = {'name': 'DepthwiseConv2D', 'kernel_size': layer.depthwise_kernel.get_shape().as_list(),
                    'strides': layer.strides, 'padding': layer.padding, 'activation': get_activation_name(layer)}

            if isinstance(layer, DepthwiseConv2DQ):
                kernel = sess.run(layer.kernel_q)
                bias = sess.run(layer.bias_q)

                if verif_preproc:
                    # precompute W*r
                    k_w, k_h, ch_in, _ = kernel.shape
                    h, w = layer.input_shape[1], layer.input_shape[2]
                    h_out, w_out = layer.output_shape[1], layer.output_shape[2]

                    np.random.seed(0)
                    r_left = np.random.randint(low=-r_max, high=r_max + 1, size=(reps, h_out * w_out)).astype(
                        np.float32)
                    w_r = np.zeros((reps, h * w, ch_in)).astype(np.float32)
                    b_r = np.zeros((reps, ch_in)).astype(np.float32)

                    X = np.zeros((1, h, w, ch_in)).astype(np.float64)
                    x_ph = tf.placeholder(tf.float64, shape=(None, h, w, ch_in))
                    w_ph = tf.placeholder(tf.float64, shape=(k_h, k_w, ch_in, 1))
                    y_ph = tf.placeholder(tf.float64, shape=(1, h_out, w_out, ch_in))
                    z = tf.nn.depthwise_conv2d_native(x_ph, w_ph, (1,) + layer.strides + (1,), layer.padding.upper())
                    dz = tf.gradients(z, x_ph, grad_ys=y_ph)[0]

                    for i in range(reps):
                        curr_r_left = r_left[i].astype(np.float64)
                        #print("r_left: {}".format(curr_r_left.astype(np.float64).sum()))
                        w_r_i = sess.run(dz, feed_dict={x_ph: X, w_ph: kernel.astype(np.float64),
                                                        y_ph: curr_r_left.reshape((1, h_out, w_out, 1)).repeat(ch_in,
                                                                                                               axis=-1)})
                        w_r[i] = np.fmod(w_r_i, p).astype(np.float32).reshape((h * w, ch_in))
                        assert np.max(np.abs(w_r[i])) < 2 ** 52
                        #print("sum(w_r) = {}".format(np.sum(w_r[i].astype(np.float64))))

                        b_r[i] = np.fmod(np.sum(curr_r_left) * bias.astype(np.float64), p)
                        #print("sum(b_r) = {}".format(np.sum(b_r[i].astype(np.float64))))

                    print("r_left: {}".format(r_left.astype(np.float64).sum()))
                    print("w_r: {}".format(w_r.astype(np.float64).sum()))
                    print("b_r: {}".format(b_r.astype(np.float64).sum()))
                    layer_weights.append(r_left.reshape(-1))
                    layer_weights.append(w_r.reshape(-1))
                    layer_weights.append(b_r.reshape(-1))

            else:
                kernel = layer.depthwise_kernel.eval(sess)
                bias = layer.bias.eval(sess)
            print("sum(abs(depthwise_w)): {}".format(np.abs(kernel).sum()))

            if not verif_preproc:
                layer_weights.append(kernel.reshape(-1).astype(dtype))
                layer_weights.append(bias.reshape(-1).astype(dtype))

        elif isinstance(layer, GlobalAveragePooling2D):
           json = {'name': 'GlobalAveragePooling2D'}

        elif isinstance(layer, Dropout):
            pass

        elif isinstance(layer, Lambda):
            pass

        elif isinstance(layer, Reshape):
            json = {'name': 'Reshape', 'shape': layer.target_shape}

        elif isinstance(layer, ZeroPadding2D):
            json = {'name': 'ZeroPadding2D',
                    'padding': layer.padding if not hasattr(layer.padding, '__len__') else layer.padding[0]}

        elif isinstance(layer, ActivationQ):
            json = {'name': 'Activation', 'type': layer.activation_name(), 'bits_w': layer.bits_w}

            if hasattr(layer, 'maxpool_params') and layer.maxpool_params is not None:
                json2 = {'name': 'MaxPooling2D', 'pool_size': layer.maxpool_params['pool_size'],
                        'strides': layer.maxpool_params['strides'], 'padding': layer.maxpool_params['padding']}
                
                json = [json, json2]

        elif isinstance(layer, ResNetBlock):
            path1 = []
            path2 = []
            for l in layer.path1:
                if isinstance(l, Conv2D) or isinstance(l, ActivationQ):
                    js, w = layer_to_json(l)
                    path1.append(js)
                    layer_weights.extend(w)

            for l in layer.path2:
                if isinstance(l, Conv2D) or isinstance(l, ActivationQ):
                    js, w = layer_to_json(l)
                    path2.append(js)
                    layer_weights.extend(w)
            
            json = {'name': 'ResNetBlock', 'identity': layer.identity, 'bits_w': layer.bits_w, 'bits_x': layer.bits_x,
                    'path1': path1, 'path2': path2}

            if slalom_privacy:
                json = [json]
                js2, _ = layer_to_json(layer.merge_act)
                if isinstance(js2, dict):
                    json.append(js2)
                else:
                    json.extend(js2)

        else:
            raise NameError("Unknown layer {}".format(layer))

        return json, layer_weights

    model_json = {'layers': [], 'shift_w': 2**bits_w, 'shift_x': 2**bits_x, 'max_tensor_size': 224*224*64}
    weights = []
    for idx, layer in enumerate(model.layers):
        json, layer_weights = layer_to_json(layer)

        if json:
            if isinstance(json, dict):
                model_json['layers'].append(json)
            else:
                model_json['layers'].extend(json)
        weights.extend(layer_weights)

    return model_json, weights


# for debugging integrity checks
def mod_test(sess, model, images, linear_ops_in, linear_ops_out, verif_preproc=False):
    linear_inputs, linear_outputs = sess.run([linear_ops_in, linear_ops_out],
                                             feed_dict={model.inputs[0]: images, keras.backend.learning_phase(): 0})

    kernels = [(layer, sess.run([layer.kernel_q, layer.bias_q])) for layer in model.layers if
               hasattr(layer, 'kernel_q')]

    p = ((1 << 24) - 3)
    r_max = (1 << 20)
    batch = images.shape[0]

    def fmod(x, p):
        return np.fmod(x, p)

    def fmod_pos(x, p):
        return np.fmod(np.fmod(x, p) + p, p)

    np.random.seed(0)
    for (layer, (kernel, bias)), inp, out in zip(kernels, linear_inputs, linear_outputs):
        assert (np.max(np.abs(inp)) < p / 2)
        assert (np.max(np.abs(out)) < p / 2)

        print("input = {} {}".format(inp.reshape(-1)[:3], inp.reshape(-1)[-3:]))
        assert np.max(np.abs(out)) < 2 ** 23

        if isinstance(layer, Conv2DQ):
            h, w = layer.input_shape[1], layer.input_shape[2]
            h_out, w_out = layer.output_shape[1], layer.output_shape[2]
            k_w, k_h, ch_in, _ = kernel.shape
            inp = np.reshape(inp, (batch, h, w, ch_in))

            pointwise = k_h == 1 and k_w == 1

            np.random.seed(0)
            if verif_preproc and pointwise:
                r_left = []
                r_right = np.random.randint(low=-r_max, high=r_max + 1, size=(2, kernel.shape[3])).astype(np.float32)
                r_right = r_right[0, :].astype(np.float64)
            else:
                r_left = np.random.randint(low=-r_max, high=r_max + 1, size=(2, h_out * w_out)).astype(np.float32)
                r_right = np.random.randint(low=-r_max, high=r_max + 1, size=(2, kernel.shape[3])).astype(np.float32)
                r_left = r_left[0, :].astype(np.float64)
                r_right = r_right[0, :].astype(np.float64)

            if batch > 1:
                r_left = np.ones(shape=(1, batch)).astype(np.float64)
                r_right = np.ones(shape=(kernel.shape[3], 1)).astype(np.float64)

            if batch > 1:
                Z = out.reshape((batch, -1)).astype(np.float64)
                r_Z = fmod(r_left.dot(Z), p).reshape(1, h_out, h_out, kernel.shape[3])
                assert np.max(np.abs(r_Z)) < 2 ** 52
                r_Z_r = r_Z.dot(r_right)
                assert np.max(np.abs(r_Z_r)) < 2 ** 52
                r_Z_r = fmod(r_Z_r, p)
                b_r = fmod(np.sum(r_left) * fmod(np.dot(bias, r_right), p), p)
                print("b_r: {}".format(b_r))
                r_Z_r = fmod_pos(r_Z_r - b_r, p)
                print(r_Z_r.shape)
            else:
                Z = out.reshape((h_out * w_out, -1)).astype(np.float64)
                Z_r = fmod(Z.dot(r_right.astype(np.float64)), p)
                assert np.max(np.abs(Z_r)) < 2 ** 52

                if not pointwise:
                    r_Z_r = r_left.dot(Z_r)
                    assert np.max(np.abs(r_Z_r)) < 2 ** 52
                    r_Z_r = fmod_pos(r_Z_r, p)
                else:
                    r_Z_r = fmod_pos(Z_r, p)
            print("r_Z_r = {} {}".format(r_Z_r.reshape(-1)[:3], r_Z_r.reshape(-1)[-3:]))

            if batch > 1:
                r_X = r_left.dot(inp.reshape(batch, -1)).reshape(1, h, h, kernel.shape[2])
                assert np.max(np.abs(r_X)) < 2 ** 52
                r_X = fmod(r_X, p)
                print("r_X = {} {}".format(r_X.reshape(-1)[:3], r_X.reshape(-1)[-3:]))
                w_right = kernel.astype(np.float64).reshape((-1, kernel.shape[-1])).dot(r_right)
                w_right = w_right.reshape(kernel.shape[0], kernel.shape[1], kernel.shape[2], 1)
                assert np.max(np.abs(w_right)) < 2 ** 52
                X_W_r = sess.run(tf.nn.conv2d(r_X, w_right, (1,) + layer.strides + (1,), layer.padding.upper()))
                assert np.max(np.abs(X_W_r)) < 2 ** 52
                X_W_r = fmod_pos(X_W_r, p)
                print(X_W_r.shape)
            else:
                if pointwise:
                    test = inp.reshape((h * w, -1)).astype(np.float64).dot(kernel.reshape(ch_in, -1).astype(np.float64))
                    test += bias.astype(np.float64)
                    print("test = {} {}".format(test.reshape(-1)[:3], test.reshape(-1)[-3:]))
                    print("out = {} {}".format(out.reshape(-1)[:3], out.reshape(-1)[-3:]))
                    assert((test.reshape(-1) == out.reshape(-1)).all())

                    w_right = kernel.astype(np.float64).reshape((-1, kernel.shape[-1])).dot(r_right.astype(np.float64))
                    assert np.max(np.abs(w_right)) < 2 ** 52
                    print("W_r = {} {}".format(w_right.reshape(-1)[:3], w_right.reshape(-1)[-3:]))

                    X_r = inp.reshape((h * w, -1)).astype(np.float64).dot(np.fmod(w_right, p))
                    assert np.max(np.abs(X_r)) < 2 ** 52

                    X_r = np.fmod(X_r, p)
                    b_r = np.fmod(np.dot(bias, r_right), p)
                    X_W_r = fmod_pos(X_r + b_r, p)

                else:
                    w_right = kernel.astype(np.float64).reshape((-1, kernel.shape[-1])).dot(r_right).reshape(-1, kernel.shape[2])
                    assert np.max(np.abs(w_right)) < 2 ** 52
                    print("sum(w_right) = {}".format(np.sum(w_right.astype(np.float64))))

                    X = np.zeros((1, h, w, ch_in)).astype(np.float64)

                    x_ph = tf.placeholder(tf.float64, shape=(None, h, w, ch_in))
                    w_ph = tf.placeholder(tf.float64, shape=(k_h, k_w, ch_in, 1))
                    y_ph = tf.placeholder(tf.float64, shape=(1, h_out, w_out, 1))
                    z = tf.nn.conv2d(x_ph, w_ph, (1,) + layer.strides + (1,), layer.padding.upper())
                    dz = tf.gradients(z, x_ph, grad_ys=y_ph)[0]

                    w_r = sess.run(dz, feed_dict={x_ph: X, w_ph: w_right.reshape(k_w, k_h, ch_in, 1),
                                                  y_ph: r_left.reshape((1, h_out, w_out, 1))})

                    assert np.max(np.abs(w_r)) < 2 ** 52

                    w_r = np.fmod(w_r, p).astype(np.float32)
                    print("sum(r_left) = {}".format(np.sum(r_left.astype(np.float64))))
                    print("sum(r_right) = {}".format(np.sum(r_right.astype(np.float64))))
                    print("sum(w_r) = {}".format(np.sum(w_r.astype(np.float64))))

                    X_W_r = inp.astype(np.float64).reshape(-1).dot(w_r.astype(np.float64).reshape(-1))
                    assert np.max(np.abs(X_W_r)) < 2 ** 52
                    X_W_r = fmod(X_W_r, p)
                    b_r = np.fmod(np.sum(r_left) * np.fmod(np.dot(bias, r_right), p), p)
                    print("b_r: {}".format(b_r))
                    X_W_r = fmod_pos(X_W_r + b_r, p)
            print("X*W_r = {} {}".format(X_W_r.reshape(-1)[:3], X_W_r.reshape(-1)[-3:]))
            if not (X_W_r == r_Z_r).all():
                np.set_printoptions(threshold=np.nan)
                print(X_W_r - r_Z_r)
                assert(0)
            print()

        elif isinstance(layer, DepthwiseConv2DQ):
            h, w = layer.input_shape[1], layer.input_shape[2]
            h_out, w_out = layer.output_shape[1], layer.output_shape[2]
            k_w, k_h, ch_in, _ = kernel.shape
            inp = np.reshape(inp, (batch, h, w, ch_in))

            reps = 1

            if batch > 1:
                r_left = np.ones(shape=(reps, batch)).astype(np.float64)

                Z = out.reshape((batch, -1)).astype(np.float64)
                r_Z = fmod(r_left.dot(Z), p).reshape(reps, h_out, h_out, ch_in)
                assert np.max(np.abs(r_Z)) < 2 ** 52
                b_r = np.sum(r_left, axis=1).reshape(reps, 1) * bias.reshape(1, ch_in)
                r_Z = fmod_pos(r_Z - b_r.reshape(reps, 1, 1, ch_in), p)
                print("r_Z = {} {}".format(r_Z.reshape(-1)[:3], r_Z.reshape(-1)[-3:]))

                r_X = r_left.dot(inp.reshape(batch, -1)).reshape(reps, h, w, ch_in)
                assert np.max(np.abs(r_X)) < 2 ** 52
                r_X = fmod(r_X, p)
                print("r_X = {} {}".format(r_X.reshape(-1)[:3], r_X.reshape(-1)[-3:]))

                r_X_W = sess.run(tf.nn.depthwise_conv2d_native(r_X, kernel, (1,) + layer.strides + (1,), layer.padding.upper()))
                assert np.max(np.abs(r_X_W)) < 2 ** 52
                r_X_W = fmod_pos(r_X_W, p)
                print("r_X_W = {} {}".format(r_X_W.reshape(-1)[:3], r_X_W.reshape(-1)[-3:]))
                assert (r_X_W == r_Z).all()
                print()

            else:
                np.random.seed(0)
                r_left = np.random.randint(low=-r_max, high=r_max + 1, size=(2, h_out * w_out)).astype(np.float32)
                r_left = r_left[0, :].astype(np.float64)

                X = np.zeros((1, h, w, ch_in)).astype(np.float64)
                x_ph = tf.placeholder(tf.float64, shape=(None, h, w, ch_in))
                w_ph = tf.placeholder(tf.float64, shape=(k_h, k_w, ch_in, 1))
                y_ph = tf.placeholder(tf.float64, shape=(1, h_out, w_out, ch_in))
                z = tf.nn.depthwise_conv2d_native(x_ph, w_ph, (1,) + layer.strides + (1,), layer.padding.upper())
                dz = tf.gradients(z, x_ph, grad_ys=y_ph)[0]

                w_r = sess.run(dz, feed_dict={x_ph: X, w_ph: kernel.astype(np.float64),
                                              y_ph: r_left.reshape((1, h_out, w_out, 1)).repeat(ch_in, axis=-1)})
                w_r = np.fmod(w_r, p).astype(np.float32).reshape((h * w, ch_in))
                assert np.max(np.abs(w_r)) < 2 ** 52
                print("sum(w_r) = {}".format(np.sum(w_r.astype(np.float64))))
                print("r_left: {}".format(r_left.astype(np.float64).sum()))

                Z_r = r_left.dot(out.reshape(h_out * w_out, -1).astype(np.float64))
                assert np.max(np.abs(Z_r)) < 2 ** 52
                Z_r = fmod_pos(Z_r, p)
                print("Z_r = {} {}".format(Z_r.reshape(-1)[:3], Z_r.reshape(-1)[-3:]))

                r_X_W = (inp.astype(np.float64).reshape(h*w, ch_in) * w_r.astype(np.float64)).sum(axis=0)
                assert np.max(np.abs(r_X_W)) < 2 ** 52
                r_X_W = fmod(r_X_W, p)

                b_r = fmod(np.sum(r_left) * bias.astype(np.float64), p)
                print("sum(b_r): {}".format(np.sum(b_r)))
                r_X_W = fmod_pos(r_X_W + b_r, p)
                print("r_X_W = {} {}".format(r_X_W.reshape(-1)[:3], r_X_W.reshape(-1)[-3:]))
                assert (r_X_W == Z_r).all()
                print()
