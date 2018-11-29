
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import python.slalom.keras_fix

import sys
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from keras import backend

from python import imagenet
from python.slalom.models import get_model
from python.slalom.quant_layers import transform, build_blinding_ops, prepare_blinding_factors, get_all_linear_layers
from python.slalom.utils import Results
from python.slalom.sgxdnn import model_to_json, SGXDNNUtils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_USE_DEEP_CONV2D"] = '0'


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.set_random_seed(1234)

    with tf.Graph().as_default():
        num_batches = args.max_num_batches
        batch_size = args.batch_size

        device = '/gpu:0'
        config = tf.ConfigProto(log_device_placement=False)
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.90
        config.gpu_options.allow_growth = True

        quantize = True
        slalom = not args.no_slalom
        blinded = args.blinding 
        integrity = args.integrity
        simulate = args.simulate

        with tf.Session(config=config) as sess:

            with tf.device(device):
                model, model_info = get_model(args.model_name, batch_size, include_top=True, double_prec=False)
            
            dataset_images, labels = imagenet.load_validation(
                args.input_dir, batch_size, preprocess=model_info['preprocess'])

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            #sgxutils = SGXDNNUtils(args.use_sgx, num_enclaves=batch_size)
            sgxutils = SGXDNNUtils(args.use_sgx, num_enclaves=1)

            num_linear_layers = len(get_all_linear_layers(model))
            if blinded and not simulate:
                queues = [tf.FIFOQueue(capacity=num_batches + 1, dtypes=[tf.float32]) for _ in range(num_linear_layers)]
            else:
                queues = None

            model, linear_ops_in, linear_ops_out = transform(model, log=False, quantize=quantize, verif_preproc=True,
                                                             slalom=slalom, slalom_integrity=integrity, slalom_privacy=blinded,
                                                             bits_w=model_info['bits_w'],
                                                             bits_x=model_info['bits_x'],
                                                             sgxutils=sgxutils, queues=queues)

            dtype = np.float32
            model_json, weights = model_to_json(sess, model, dtype=dtype, verif_preproc=True, slalom_privacy=blinded,
                                                bits_w=model_info['bits_w'], bits_x=model_info['bits_x'])
            sgxutils.load_model(model_json, weights, dtype=dtype, verify=True, verify_preproc=True)

            num_classes = np.prod(model.output.get_shape().as_list()[1:])
            print("num_classes: {}".format(num_classes))

            print_acc = (num_classes == 1000)
            res = Results(acc=print_acc)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            sgxutils.slalom_init(integrity, (blinded and not simulate), batch_size)
            if blinded and not simulate:
                in_ph, zs, out_ph, queue_ops, temps, out_funcs = build_blinding_ops(model, queues, batch_size)

            for i in range(num_batches):
                images, true_labels = sess.run([dataset_images, labels])

                if quantize:
                    images = np.round(2 ** model_info['bits_x'] * images).astype(np.float32)
                    print("input images: {}".format(np.sum(np.abs(images))))

                if blinded and not simulate:
                    prepare_blinding_factors(sess, model, sgxutils, in_ph, zs, out_ph, queue_ops, batch_size, num_batches=1,
                                             #inputs=images, temps=temps, out_funcs=out_funcs
                    )

                images = sgxutils.slalom_blind_input(images)
                print("blinded images: {}".format((np.min(images), np.max(images), np.sum(np.abs(images.astype(np.float64))))))
                print(images.reshape(-1)[:3], images.reshape(-1)[-3:])

                res.start_timer()

                preds = sess.run(model.outputs[0], feed_dict={model.inputs[0]: images,
                                                              backend.learning_phase(): 0},
                                 options=run_options, run_metadata=run_metadata)

                preds = np.reshape(preds, (batch_size, -1))
                res.end_timer(size=len(images))
                res.record_acc(preds, true_labels)
                res.print_results()
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open('timeline_{}_{}.json'.format(args.model_name, device[1:4]), 'w') as f:
                    f.write(ctf)

                sys.stdout.flush()
            coord.request_stop()
            coord.join(threads)

        if sgxutils is not None:
            sgxutils.destroy()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', type=str,
                        choices=['vgg_16', 'vgg_19', 'inception_v3', 'mobilenet', 'mobilenet_sep', 
                                 'resnet_18', 'resnet_34', 'resnet_50', 'resnet_101', 'resnet_152'])
    parser.add_argument('--input_dir', type=str,
                        default='../imagenet/',
                        help='Input directory with images.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='How many images process at one time.')
    parser.add_argument('--max_num_batches', type=int, default=2,
                        help='Max number of batches to evaluate.')
    parser.add_argument('--use_sgx', action='store_true')
    parser.add_argument('--no_slalom', action='store_true', help='only test GPU quantization')
    parser.add_argument('--blinding', action='store_true', help='add random blinding for privacy')
    parser.add_argument('--integrity', action='store_true', help='add integrity checks')
    parser.add_argument('--simulate', action='store_true')
    args = parser.parse_args()

    tf.app.run()
