
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from keras import backend

from python import imagenet
from python.slalom.models import get_model
from python.slalom.quant_layers import transform
from python.slalom.utils import Results, timer
from python.slalom.sgxdnn import model_to_json, SGXDNNUtils, mod_test

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_USE_DEEP_CONV2D"] = '0'

DTYPE_VERIFY = np.float32


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.set_random_seed(1234)

    with tf.Graph().as_default():
        # Prepare graph
        num_batches = args.max_num_batches

        sgxutils = None

        if args.mode == 'tf-gpu':
            assert not args.use_sgx

            device = '/gpu:0'
            config = tf.ConfigProto(log_device_placement=False)
            config.allow_soft_placement = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.90
            config.gpu_options.allow_growth = True

        elif args.mode == 'tf-cpu':
            assert not args.verify and not args.use_sgx

            device = '/cpu:0'
            #config = tf.ConfigProto(log_device_placement=False)
            config = tf.ConfigProto(log_device_placement=False, device_count={'CPU': 1, 'GPU': 0})
            config.intra_op_parallelism_threads = 1
            config.inter_op_parallelism_threads = 1

        else:
            assert args.mode == 'sgxdnn'

            device = '/gpu:0'
            config = tf.ConfigProto(log_device_placement=False)
            config.allow_soft_placement = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
            config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            with tf.device(device):
                model, model_info = get_model(args.model_name, args.batch_size, include_top=not args.no_top)

            dataset_images, labels = imagenet.load_validation(args.input_dir, args.batch_size,
                                                              preprocess=model_info['preprocess'],
                                                              num_preprocessing_threads=1)

            model, linear_ops_in, linear_ops_out = transform(model, log=False, quantize=args.verify,
                                                             verif_preproc=args.preproc,
                                                             bits_w=model_info['bits_w'],
                                                             bits_x=model_info['bits_x'])

            if args.mode == 'sgxdnn':
                #sgxutils = SGXDNNUtils(args.use_sgx, num_enclaves=args.batch_size)
                #sgxutils = SGXDNNUtils(args.use_sgx, num_enclaves=2)
                sgxutils = SGXDNNUtils(args.use_sgx)

                dtype = np.float32 if not args.verify else DTYPE_VERIFY
                model_json, weights = model_to_json(sess, model, args.preproc, dtype=dtype,
                                                    bits_w=model_info['bits_w'], bits_x=model_info['bits_x'])
                sgxutils.load_model(model_json, weights, dtype=dtype, verify=args.verify, verify_preproc=args.preproc)

            num_classes = np.prod(model.output.get_shape().as_list()[1:])
            print("num_classes: {}".format(num_classes))
            
            print_acc = (num_classes == 1000)
            res = Results(acc=print_acc)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            #from multiprocessing.dummy import Pool as ThreadPool
            #pool = ThreadPool(3)
    
            for i in range(num_batches):
                images, true_labels = sess.run([dataset_images, labels])

                if args.verify:
                    images = np.round(2**model_info['bits_x'] * images)
                print("input images: {}".format(np.sum(np.abs(images))))

                if args.mode in ['tf-gpu', 'tf-cpu']:
                    res.start_timer()
                    preds = sess.run(model.outputs[0], feed_dict={model.inputs[0]: images,
                                                                  backend.learning_phase(): 0},
                                     options=run_options, run_metadata=run_metadata)

                    print(np.sum(np.abs(images)), np.sum(np.abs(preds)))
                    preds = np.reshape(preds, (args.batch_size, num_classes))
                    res.end_timer(size=len(images))
                    res.record_acc(preds, true_labels)
                    res.print_results()

                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)

                else:
                    res.start_timer()
                    
                    if args.verify:
                       
                        t1 = timer() 
                        linear_outputs = sess.run(linear_ops_out, feed_dict={model.inputs[0]: images,
                                                                             backend.learning_phase(): 0},
                                                  options=run_options, run_metadata=run_metadata)
                        t2 = timer()
                        print("GPU compute time: {:.4f}".format((t2-t1) / (1.0 * args.batch_size)))

                        #mod_test(sess, model, images, linear_ops_in, linear_ops_out, verif_preproc=args.preproc)

                        def func(data):
                            return sgxutils.predict_and_verify(data[1], data[2], num_classes=num_classes, dtype=dtype, eid_idx=0)

                        if not args.verify_batched:
                            start = timer()

                            linear_outputs_batched = [x.reshape((args.batch_size, -1)) for x in linear_outputs]

                            preds = []
                            for i in range(args.batch_size):
                                t1 = timer()
                                aux_data = [x[i] for x in linear_outputs_batched]
                                pred = sgxutils.predict_and_verify(images[i:i+1], aux_data,
                                                                  num_classes=num_classes, dtype=dtype)
                                t2 = timer()
                                print("verify time: {:.4f}".format((t2-t1)))
                                preds.append(pred)
                            preds = np.vstack(preds)
                            end = timer()
                            print("avg verify time: {:.4f}".format((end-start) / (1.0 * args.batch_size)))

                            #all_data = [(i, images[i:i+1], [x[i] for x in linear_outputs_batched]) for i in range(args.batch_size)]
                            #preds = np.vstack(pool.map(func, all_data))
                        else:
                            preds = sgxutils.predict_and_verify(images, linear_outputs, num_classes=num_classes, dtype=dtype)

                    else:

                        def func(data):
                            return sgxutils.predict(data[1], num_classes=num_classes, eid_idx=0)

                        #all_data = [(i, images[i:i+1]) for i in range(args.batch_size)]
                        #preds = np.vstack(pool.map(func, all_data))

                        preds = []
                        for i in range(args.batch_size):
                            pred = sgxutils.predict(images[i:i + 1], num_classes=num_classes)
                            preds.append(pred)
                        preds = np.vstack(preds)

                    res.end_timer(size=len(images))
                    res.record_acc(preds, true_labels)
                    res.print_results()

                    
                    #tl = timeline.Timeline(run_metadata.step_stats)
                    #ctf = tl.generate_chrome_trace_format()
                    #ctf_j = json.loads(ctf)
                    #events = [e["ts"] for e in ctf_j["traceEvents"] if "ts" in e]
                    #print("TF Timeline: {:.4f}".format((np.max(events) - np.min(events)) / (1000000.0 * args.batch_size)))

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

    parser.add_argument('mode', type=str, choices=['tf-gpu', 'tf-cpu', 'sgxdnn'])

    parser.add_argument('--input_dir', type=str,
                        default='../imagenet/',
                        help='Input directory with images.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='How many images process at one time.')
    parser.add_argument('--max_num_batches', type=int, default=2,
                        help='Max number of batches to evaluate.')
    parser.add_argument('--verify', action='store_true',
                        help='Activate verification.')
    parser.add_argument('--preproc', action='store_true',
                        help='Use preprocessing for verification.')
    parser.add_argument('--use_sgx', action='store_true')
    parser.add_argument('--verify_batched', action='store_true',
                        help='Use batched verification.')
    parser.add_argument('--no_top', action='store_true',
                        help='Omit top part of network.')
    args = parser.parse_args()

    tf.app.run()
