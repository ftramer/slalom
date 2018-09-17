import tensorflow as tf
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.mobilenet import MobileNet
from keras.layers import Input
from python.slalom.utils import preprocess_vgg, print_model_size
from python.preprocessing.preprocessing_factory import get_preprocessing
from python.slalom.mobilenet_sep import MobileNet_sep
import numpy as np


def get_model(model_name, batch_size, include_top=True, double_prec=False):
    if model_name in ['vgg_16']:
        h = w = 224
        assert(h % 2**5 == 0)
        num_classes = 1000
        model_func = lambda weights, inp: VGG16(include_top=include_top, weights=weights, input_tensor=inp, input_shape=(h, w, 3), pooling=None, classes=num_classes)
        preprocess_func = preprocess_vgg
        bits_w = 8
        bits_x = 0
    elif model_name in ['vgg_19']:
        h = w = 224
        num_classes = 1000
        model_func = lambda weights, inp: VGG19(include_top=include_top, weights=weights, input_tensor=inp, input_shape=(h, w, 3), pooling=None, classes=num_classes)
        preprocess_func = preprocess_vgg
        bits_w = 8
        bits_x = 0
    elif model_name in ['mobilenet']:
        h = w = 224
        num_classes = 1000
        model_func = lambda weights, inp: MobileNet(include_top=include_top, weights=weights, input_tensor=inp, input_shape=(224, 224, 3), pooling=None, classes=num_classes)
        preprocess_func = get_preprocessing('mobilenet_v1')
        bits_w = 8
        bits_x = 8
    elif model_name in ['mobilenet_sep']:
        h = w = 224
        num_classes = 1000
        model_func = lambda weights, inp: MobileNet_sep(include_top=include_top, input_tensor=inp, input_shape=(224, 224, 3), pooling=None, classes=num_classes)
        preprocess_func = get_preprocessing('mobilenet_v1')
        bits_w = 8
        bits_x = 8 
    else:
        raise AttributeError("unknown model {}".format(model_name))

    images = tf.placeholder(dtype=tf.float32, shape=(batch_size, h, w, 3))
    model = model_func("imagenet", images)
    preprocess = lambda x: preprocess_func(x, h, w)

    if double_prec:
        images_dbl = tf.placeholder(dtype=tf.float64, shape=(batch_size, h, w, 3))
        images_dbl = Input(images_dbl, (batch_size, h, w, 3), dtype=tf.float64)
        K.set_floatx('float64')
        model_dbl = model_func(None, images_dbl)
        
        for i in range(len(model_dbl.layers)):
            weights = model.layers[i].get_weights()
            weights_dbl = [None if w is None else w.astype(np.float64) for w in weights]
            model_dbl.layers[i].set_weights(weights_dbl)
        
        model = model_dbl
        preprocess = lambda x: preprocess_func(x, h, w, dtype=tf.float64)
        K.set_floatx('float32')


    print_model_size(model)
    
    res = {}
    res['preprocess'] = preprocess
    res['bits_w'] = bits_w
    res['bits_x'] = bits_x
    return model, res
