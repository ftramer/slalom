import numpy as np
import tensorflow as tf

from slalom import imagenet
from slalom.utils import Results
from slalom.quant import Conv2DQ, DenseQ
from preprocessing.vgg_preprocessing import \
    _aspect_preserving_resize, _central_crop, _RESIZE_SIDE_MIN
from preprocessing.preprocessing_factory import get_preprocessing
import sys

from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Conv2D, Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import clone_model
from keras import Model
import keras.backend as K


def preprocess_vgg(image):
    image = _aspect_preserving_resize(image, _RESIZE_SIDE_MIN)
    image = _central_crop([image], 224, 224)[0]
    image.set_shape([224, 224, 3])
    image = tf.to_float(image)
    return preprocess_input(image)


def size_to_mb(s, type_bytes=4):
    return (type_bytes * s) / (1.0 * 1024**2)


def quantize(model, bits_w, bits_x):
    layer_map = {}

    for i, layer in enumerate(model.layers):
        if isinstance(layer, Conv2D):
            if (i < len(model.layers) -1) and isinstance(layer.outbound_nodes[0].outbound_layer, BatchNormalization):
                assert not layer.use_bias
                layer.add_weight(shape=(layer.filters,),
                                initializer=layer.bias_initializer,
                                name='bias',
                                regularizer=layer.bias_regularizer,
                                constraint=layer.bias_constraint)
                layer.use_bias = True

            conf = layer.get_config()
            conf['bits_w'] = bits_w
            conf['bits_x'] = bits_x
            layer_map[layer] = Conv2DQ.from_config(conf)
        elif isinstance(layer, Dense):
            conf = layer.get_config()
            conf['bits_w'] = bits_w
            conf['bits_x'] = bits_x
            if conf.get('activation') == 'softmax':
                del conf['activation']
            layer_map[layer] = DenseQ.from_config(conf)
        elif isinstance(layer, BatchNormalization):
            conv = layer.inbound_nodes[0].inbound_layers[0]
            assert isinstance(conv, Conv2D)
            assert layer.axis == 3

            mean = layer.moving_mean
            var = layer.moving_variance
            beta = layer.beta if layer.beta is not None else 0.0
            gamma = layer.gamma if layer.gamma is not None else 1.0

            w = conv.get_weights()[0]
            
            new_w = K.get_session().run(w * gamma / (K.sqrt(var) + layer.epsilon))
            new_b = K.get_session().run(-(mean*gamma) / (K.sqrt(var) + layer.epsilon) + beta)
            conv.set_weights((new_w, new_b))
            layer_map[layer] = Lambda(lambda x: x)

    depth_keys = list(model.nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    for depth in depth_keys:
        nodes = model.nodes_by_depth[depth]
        for node in nodes:
            if node.outbound_layer in layer_map:
                node.outbound_layer = layer_map[node.outbound_layer]

    result_model = clone_model(model)   
    for i in range(len(result_model.layers)):
        src = model.layers[i]
        dest = result_model.layers[i]

        if isinstance(src, Conv2D) or isinstance(src, Dense):
            assert(isinstance(dest, Conv2DQ) or isinstance(dest, DenseQ))
            dest.set_weights(src.get_weights())

    return result_model


def print_model_size(model):
    tot_size = 0.0

    for layer in model.layers:
        print(layer.name)
        if layer.__class__ in [Conv2D, Dense]:
            layer_size = np.prod(layer.output.get_shape().as_list()[1:])
            tot_size += layer_size
            print("Layer {}: {:.4f} MB".format(layer.name, size_to_mb(layer_size)))

    print("Total Size: {:.2f} MB".format(size_to_mb(tot_size)))


def slow_verify_linear_comps(linear_outputs):
    for i in range(len(linear_outputs)):
        name, X, W, Z = linear_outputs[i]
        if len(X.shape) == 4:
            #print("Conv2D with input {} and kernel {} to output {}".format(X.shape, W.shape, Z.shape))
            #X.astype(np.int32)
            #W = W.astype(np.int32).reshape((-1, W.shape[-1]))
            #Z = Z.astype(np.int32).reshape((-1, Z.shape[-1]))
            W = W.reshape((-1, W.shape[-1]))
            Z = Z.reshape((-1, Z.shape[-1]))
            X_patches = tf.extract_image_patches(X, 
                                                 ksizes=[1, 3, 3, 1], 
                                                 strides=[1, 1, 1, 1], 
                                                 rates=[1, 1, 1, 1], 
                                                 padding='SAME')
            
            Z2 = K.get_session().run(tf.matmul(tf.reshape(X_patches, (Z.shape[0], -1)), W))
            norm1 = np.linalg.norm(Z)
            norm2 = np.linalg.norm(Z2)
            norm3 = np.linalg.norm(Z - Z2)
            eq = np.array_equal(Z, Z2)
            print("{:.3f}, {:.3f}, {:.3f}, {}".format(norm1, norm2, norm3, eq))
	elif len(X.shape) == 2:
            #print("MatMul with input {} and kernel {} to output {}".format(X.shape, W.shape, Z.shape))
            #X = X.astype(np.int32)
            #W = W.astype(np.int32)
            #Z = Z.astype(np.int32)
            
            Z2 = K.get_session().run(tf.matmul(X, W))
            norm1 = np.linalg.norm(Z)
            norm2 = np.linalg.norm(Z2)
            norm3 = np.linalg.norm(Z - Z2)
            eq = np.array_equal(Z, Z2)
            print("{:.3f}, {:.3f}, {:.3f}, {}".format(norm1, norm2, norm3, eq))
        else:
            raise ArgumentError("linear input has shape {}".format(X.shape))


def test_forward(model, preprocess, quant=False, bits_w=8, bits_x=8, verify=False):

    assert(not verify or quant)
    old_ops = K.get_session().graph.get_operations()
    linear_ops = K.constant(0)

    if quant:
       print("quantizing model...")
       model = quantize(model, bits_w=bits_w, bits_x=bits_x)
       print("quantization completed!")
       if verify:
           new_ops = [op for op in K.get_session().graph.get_operations() if op not in old_ops]
           linear_ops = [(op.name, op.inputs[0], op.inputs[1], op.outputs[0]) for op in new_ops if op.type in ['Conv2D', 'MatMul']]

    dataset_images, labels = imagenet.load_validation(
        args.input_dir, args.batch_size, preprocess=preprocess)

    num_batches = args.max_num_batches

    sess = K.get_session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    res = Results()
    res_verif = Results(acc=False)

    for i in range(num_batches):
        images, true_labels = sess.run([dataset_images, labels])

        if quant:
            images = np.round(2**bits_x * images)

        res.start_timer()
        preds, linear_outputs = sess.run([model.output, linear_ops], feed_dict={model.inputs[0]: images, K.learning_phase(): 0})
        res.end_timer()
        res.record_acc(preds, true_labels)
        res.print_results()

        if verify:
            res_verif.start_timer()
            slow_verify_linear_comps(linear_outputs) 
            res_verif.end_timer(size=args.batch_size)
            res_verif.print_results()

        sys.stdout.flush()

    coord.request_stop()
    coord.join(threads)


def main(_):

        if args.model_name in ['vgg_16']:
            images = tf.placeholder(dtype=tf.float32, shape=(args.batch_size, 224, 224, 3))
            num_classes = 1000
            model = VGG16(include_top=True, weights='imagenet', input_tensor=images, input_shape=None, pooling=None, classes=num_classes)
            preprocess = preprocess_vgg
            bits_w = 8
            bits_x = 2
        elif args.model_name in ['inception_v3']:
            images = tf.placeholder(dtype=tf.float32, shape=(args.batch_size, 299, 299, 3))
            num_classes = 1000
            model = InceptionV3(include_top=True, weights='imagenet', input_tensor=images, input_shape=None, pooling=None, classes=num_classes)
            preprocess = lambda x: get_preprocessing('inception_v3')(x, 299, 299)
            bits_w = 8
            bits_x = 8
        else:
            raise AttributeError("unknown model {}".format(args.model_name))

        if args.test_name == 'model_size':
            print_model_size(model)
        elif args.test_name == 'forward':
            test_forward(model, preprocess, quant=args.quant, bits_w=bits_w, bits_x=bits_x, verify=args.verify)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('test_name', type=str,
                        choices=['model_size', 'forward', 'backward'])
    parser.add_argument('model_name', type=str,
                        choices=['vgg_16', 'inception_v3'])

    parser.add_argument('--input_dir', type=str,
                        default='/home/ubuntu/imagenet/',
                        help='Input directory with images.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='How many images process at one time.')
    parser.add_argument('--max_num_batches', type=int, default=2,
                        help='Max number of batches to evaluate.')
    parser.add_argument('--quant', action='store_true',
                        help='Activate quantization.')
    parser.add_argument('--verify', action='store_true',
                        help='Activate verification.')
    args = parser.parse_args()
    tf.app.run()
