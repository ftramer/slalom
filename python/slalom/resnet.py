# -*- coding: utf-8 -*-
"""ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras.layers import *
from keras import layers
from keras.models import Model, Sequential
from keras import backend as K
from keras.engine import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import load_weights_from_hdf5_group_by_name, h5py
import tensorflow as tf

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


class ResNetBlock(Layer):
    def __init__(self, kernel_size, filters, stage, block, identity=False, strides=(2, 2),
                 path1=None, path2=None, merge_act=None, quantize=False, bits_w=8, bits_x=8,
                 slalom=False, slalom_integrity=False, slalom_privacy=False, use_bn=False, basic=False, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)

        self.quantize = quantize
        self.slalom = slalom
        self.slalom_integrity = slalom_integrity
        self.slalom_privacy = slalom_privacy
        self.bits_w = bits_w
        self.bits_x = bits_x
        self.range_w = 2 ** bits_w
        self.range_x = 2 ** bits_x

        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.identity = identity

        self.basic = basic
        self.stage = stage
        self.block = block
        self.conv_name_base = 'res' + str(stage) + block + '_branch'
        self.bn_name_base = 'bn' + str(stage) + block + '_branch'
        self.use_bn = use_bn

        self.path1 = [] if path1 is None else path1 
        self.path2 = [] if path2 is None else path2 
        self.merge_act = merge_act

    def create_layers(self, input_shape):
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        if self.basic:
            self.create_layers_basic(input_shape)
            return

        filters1, filters2, filters3 = self.filters

        shape = input_shape
        if self.identity:
            self.path1.append(Conv2D(filters1, (1, 1), name=self.conv_name_base + '2a', input_shape=shape))
        else:
            self.path1.append(Conv2D(filters1, (1, 1), strides=self.strides, name=self.conv_name_base + '2a', input_shape=shape))

        self.path1[-1].build(shape)
        shape = self.path1[-1].compute_output_shape(shape)
        self.path1.append(BatchNormalization(axis=bn_axis, name=self.bn_name_base + '2a', input_shape=shape))
        self.path1[-1].build(shape)
        self.path1.append(Activation('relu'))
        self.path1[-1].build(shape)

        self.path1.append(Conv2D(filters2, self.kernel_size,
                                padding='same', name=self.conv_name_base + '2b', input_shape=shape))
        self.path1[-1].build(shape)
        shape = self.path1[-1].compute_output_shape(shape)

        self.path1.append(BatchNormalization(axis=bn_axis, name=self.bn_name_base + '2b', input_shape=shape))
        self.path1[-1].build(shape)
        self.path1.append(Activation('relu'))
        self.path1[-1].build(shape)

        self.path1.append(Conv2D(filters3, (1, 1), name=self.conv_name_base + '2c', input_shape=shape))
        self.path1[-1].build(shape)
        shape = self.path1[-1].compute_output_shape(shape)

        self.path1.append(BatchNormalization(axis=bn_axis, name=self.bn_name_base + '2c', input_shape=shape))
        self.path1[-1].build(shape)

        if not self.identity:
            shape = input_shape
            self.path2.append(Conv2D(filters3, (1, 1), strides=self.strides,
                                            name=self.conv_name_base + '1', input_shape=shape))
            self.path2[-1].build(shape)
            shape = self.path2[-1].compute_output_shape(shape)

            self.path2.append(BatchNormalization(axis=bn_axis, name=self.bn_name_base + '1', input_shape=shape))
            self.path2[-1].build(shape)

        self.merge_act = Activation('relu')
        self.merge_act.build(shape)

    def create_layers_basic(self, input_shape):
        filters, _, _= self.filters

        shape = input_shape
        if self.identity:
            self.path1.append(Conv2D(filters, self.kernel_size, name=self.conv_name_base + '2a', input_shape=shape, padding='same'))
        else:
            self.path1.append(Conv2D(filters, self.kernel_size, strides=self.strides, name=self.conv_name_base + '2a', input_shape=shape, padding='same'))

        self.path1[-1].build(shape)
        shape = self.path1[-1].compute_output_shape(shape)
        self.path1.append(Activation('relu'))
        self.path1[-1].build(shape)

        self.path1.append(Conv2D(filters, self.kernel_size,
                                padding='same', name=self.conv_name_base + '2b', input_shape=shape))
        self.path1[-1].build(shape)
        shape = self.path1[-1].compute_output_shape(shape)

        if not self.identity and self.strides != (1, 1):
            shape = input_shape
            self.path2.append(Conv2D(filters, (1, 1), strides=self.strides,
                                            name=self.conv_name_base + '1', input_shape=shape))
            self.path2[-1].build(shape)
            shape = self.path2[-1].compute_output_shape(shape)

        self.merge_act = Activation('relu')
        self.merge_act.build(shape)

    def build(self, input_shape):
        super(ResNetBlock, self).build(input_shape)

        if self.path1 or self.path2:
            shape = input_shape
            for l in self.path1:
                l.build(shape)
                shape = l.compute_output_shape(shape)

            shape = input_shape
            for l in self.path2:
                l.build(shape)                
                shape = l.compute_output_shape(shape)
            
            self.merge_act.build(shape)

        else:
            self.create_layers(input_shape)

    def compute_output_shape(self, input_shape):
        shape = input_shape
        for l in self.path1:
            shape = l.compute_output_shape(shape)
        return self.merge_act.compute_output_shape(shape)

    def get_layers(self):
        if self.use_bn:
            #layers = [l for l in self.path1 + self.path2 if isinstance(l, Conv2D) or isinstance(l, BatchNormalization)]
            layers = [l for l in self.path1 + self.path2]
        else:
            layers = [l for l in self.path1 + self.path2 if not isinstance(l, BatchNormalization)]
        layers.append(self.merge_act)
        return layers
        
    def call(self, inputs):
        out1 = inputs
        for l in self.path1:
            out1 = l(out1)

        out2 = inputs
        for l in self.path2:
            out2 = l(out2)

        if self.quantize and not self.path2:
            out2 *= 2**self.bits_w

        merge = out1 + out2
        merge = self.merge_act(merge)
        return merge

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'filters': self.filters,
            'strides': self.strides,
            'identity': self.identity,
            'conv_name_base': self.conv_name_base,
            'bn_name_base': self.bn_name_base,
            'basic': self.basic
        }

        base_config = super(ResNetBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None, classes=1000, layers=50):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    When using TensorFlow, for best performance you should
    set `"image_data_format": "channels_last"` in the config.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    assert layers in [18, 34, 50, 101, 152]
    use_bn = (layers == 50)
    basic = (layers in [18, 34])

    if layers == 18:
        num_layers = [2, 2, 2, 2]
    elif layers == 34:
        num_layers = [3, 4, 6, 3]
    elif layers == 50:
        num_layers = [3, 4, 6, 3]
    elif layers == 101:
        num_layers = [3, 4, 23, 3]
    elif layers == 152:
        num_layers = [3, 8, 36, 3]

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    if basic:
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = ResNetBlock(3, [64, 64, 256], stage=2, block='a', use_bn=use_bn, basic=basic)(x)
    else:
        x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
        if use_bn:
            x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = ResNetBlock(3, [64, 64, 256], stage=2, block='a', strides=(1, 1), use_bn=use_bn, basic=basic)(x)

    
    for i in range(num_layers[0] - 1):
        x = ResNetBlock(3, [64, 64, 256], stage=2, block=chr(ord('b') + i), identity=True, use_bn=use_bn, basic=basic)(x)

    x = ResNetBlock(3, [128, 128, 512], stage=3, block='a', use_bn=use_bn, basic=basic)(x)
    for i in range(num_layers[1] - 1):
        x = ResNetBlock(3, [128, 128, 512], stage=3, block=chr(ord('b') + i), identity=True, use_bn=use_bn, basic=basic)(x)

    x = ResNetBlock(3, [256, 256, 1024], stage=4, block='a', use_bn=use_bn, basic=basic)(x)
    for i in range(num_layers[2] - 1):
        x = ResNetBlock(3, [256, 256, 1024], stage=4, block=chr(ord('b') + i), identity=True, use_bn=use_bn, basic=basic)(x)

    x = ResNetBlock(3, [512, 512, 2048], stage=5, block='a', use_bn=use_bn, basic=basic)(x)
    for i in range(num_layers[3] - 1):
        x = ResNetBlock(3, [512, 512, 2048], stage=5, block=chr(ord('b') + i), identity=True, use_bn=use_bn, basic=basic)(x)

    if basic:
        x = GlobalAveragePooling2D()(x)
    else:
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet' and layers == 50:
        if include_top:
            weights_path = get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')

        with h5py.File(weights_path, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']

            import itertools
            all_layers = [[l] if not isinstance(l, ResNetBlock) else l.get_layers() for l in model.layers]
            all_layers = list(itertools.chain.from_iterable(all_layers))
            load_weights_from_hdf5_group_by_name(f, all_layers)

        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

    return model
