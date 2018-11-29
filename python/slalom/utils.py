from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from python.preprocessing.vgg_preprocessing import _aspect_preserving_resize, \
    _central_crop, _RESIZE_SIDE_MIN
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import *
from python.slalom.resnet import ResNetBlock
import tensorflow as tf
import numpy as np
from timeit import default_timer as timer
import sys
import itertools


class Results(object):

    def __init__(self, acc=True, time=True, k=5):
        self.acc = acc
        self.time = time

        self.top1_acc = 0.0
        self.topk_acc = 0.0
        self.tot_times = []
        self.start = 0
        self.pred_counts = []
        self.k = k

    def start_timer(self):
        self.start = timer()

    def end_timer(self, size=None):
        assert self.acc or (size is not None)

        self.tot_times.append(timer() - self.start)
        print("recorded time: {}".format(self.tot_times[-1]))
        if not self.acc and size is not None:
            self.pred_counts.append(size)

    def record_acc(self, preds, true_labels):
        if not self.acc:
            return

        top1, topk = get_topk_acc(preds, true_labels, k=self.k)
        self.top1_acc += top1
        self.topk_acc += topk
        self.pred_counts.append(len(preds))

    def print_results(self):
        assert len(self.pred_counts) == len(self.tot_times)
        assert len(self.pred_counts) > 0

        if self.acc:
            print("\ttop1 err: {:.1f}".format(100 * (1.0 - self.top1_acc / np.sum(self.pred_counts))))
            print("\ttop{} err: {:.1f}".format(self.k, 100 * (1.0 - self.topk_acc / np.sum(self.pred_counts))))

        if not self.time:
            return

        if len(self.tot_times) > 1:
            avg_time = np.sum(self.tot_times[1:]) / (1.0 * np.sum(self.pred_counts[1:]))
        else:
            avg_time = np.sum(self.tot_times) / (1.0 * np.sum(self.pred_counts))

        real_time = self.tot_times[-1] / (1.0 * self.pred_counts[-1])
        print("\tprocess one image per {:.4f} s ({:.4f} s realtime)".format(avg_time, real_time))
        sys.stdout.flush()


def get_topk_acc(preds, true_labels, k=5):
    batch_size = len(true_labels)
    assert(len(preds.shape) == 2)
    assert(preds.shape[0] == batch_size)

    preds = preds.argsort(axis=1)[:, -k:][:, ::-1] + 1
    return np.sum(preds[:, 0] == true_labels), \
           np.sum(np.any(preds.T == true_labels, axis=0))


def size_to_mb(s, type_bytes=4):
    return (type_bytes * s) / (1.0 * 1024**2)


def print_model_size(model):
    print("computing model size...")
    tot_params_size = 0.0
    tot_input_size = 0.0
    tot_output_size = 0.0

    all_layers = [[l] if not isinstance(l, ResNetBlock) else l.get_layers() for l in model.layers]
    all_layers = list(itertools.chain.from_iterable(all_layers))

    for layer in all_layers:
        if layer.__class__ in [Conv2D, Dense, DepthwiseConv2D]:
            layer_output_size = np.prod(layer.output_shape[1:])
            layer_input_size = np.prod(layer.input_shape[1:])
            layer_params_size = np.sum([np.prod(w.shape) for w in layer.get_weights()])
            tot_output_size += layer_output_size
            tot_input_size += layer_input_size
            tot_params_size += layer_params_size
            print("Layer {}: inp: {:.3f} MB, out: {:.3f} MB, params: {:.3f} MB"
                .format(layer.name, size_to_mb(layer_input_size), size_to_mb(layer_output_size), size_to_mb(layer_params_size)))
        else:
            if layer.__class__ not in \
                [MaxPooling2D, Flatten, InputLayer, ZeroPadding2D, BatchNormalization, Activation, GlobalAveragePooling2D, Reshape, Dropout]:
                print("Skipping layer {} ({})".format(layer.name, layer.__class__))

    print("Total Params: {:.0f} params, {:.2f} MB".format(tot_params_size, size_to_mb(tot_params_size)))
    print("Total Inputs: {:.0f} params, {:.2f} MB".format(tot_input_size, size_to_mb(tot_input_size)))
    print("Total Outputs: {:.0f} params, {:.2f} MB".format(tot_output_size, size_to_mb(tot_output_size)))


def preprocess_vgg(image, h=224, w=224, dtype=tf.float32):
    if h <= 224 and w <= 224:
        image = _aspect_preserving_resize(image, _RESIZE_SIDE_MIN)
    image = _central_crop([image], h, w)[0]
    image.set_shape([h, w, 3])
    image = tf.cast(image, dtype)
    return preprocess_input(image)


def get_all_layers(model):
    all_layers = [[l] if not isinstance(l, ResNetBlock) else l.get_layers() for l in model.layers]
    all_layers = list(itertools.chain.from_iterable(all_layers))
    return all_layers

