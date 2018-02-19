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
from keras.layers import Conv2D, Dense
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


def quantize(model):
    layers = model.layers
    x = layers[0].input
    for i in range(0, len(layers)):
        x = layers[i](x)

    # Final touch
    result_model = Model(input=layers[0].input, output=x)
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


def test_forward(sess, x, model, preprocess, quant=False):

    model.summary()
    if quant:
        model = quantize(model)
        model.summary()

    logits = model(x)

    dataset_images, labels = imagenet.load_validation(
        args.input_dir, args.batch_size, preprocess=preprocess)

    num_batches = args.max_num_batches

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    res = Results()

    for i in range(num_batches):
        images, true_labels = sess.run([dataset_images, labels])

        res.start_timer()
        preds = sess.run(logits, feed_dict={x: images, K.learning_phase(): 0})
        res.end_timer()

        res.record_acc(preds, true_labels)
        res.print_results()
        sys.stdout.flush()

    coord.request_stop()
    coord.join(threads)


def main(_):

    with tf.Session() as sess:

        if args.model_name in ['vgg_16']:
            images = tf.placeholder(dtype=tf.float32, shape=(args.batch_size, 224, 224, 3))
            num_classes = 1000
            model = VGG16(include_top=True, weights='imagenet', input_tensor=images, input_shape=None, pooling=None, classes=num_classes)
            preprocess = preprocess_vgg
        elif args.model_name in ['inception_v3']:
            images = tf.placeholder(dtype=tf.float32, shape=(args.batch_size, 299, 299, 3))
            num_classes = 1000
            model = InceptionV3(include_top=True, weights='imagenet', input_tensor=images, input_shape=None, pooling=None, classes=num_classes)
            preprocess = lambda x: get_preprocessing('inception_v3')(x, 299, 299)
        else:
            raise AttributeError("unknown model {}".format(args.model_name))

        if args.test_name == 'model_size':
            print_model_size(model)
        elif args.test_name == 'forward':
            test_forward(sess, images, model, preprocess)


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
    args = parser.parse_args()
    tf.app.run()
