import numpy as np
import tensorflow as tf

from nets import nets_factory
from preprocessing import preprocessing_factory
from slalom import imagenet
from slalom.utils import Results
import sys


def size_to_mb(s, type_bytes=4):
    return (type_bytes * s) / (1.0 * 1024**2)


def print_model_size(sess):
    tot_size = 0.0
    ops = sess.graph.get_operations()
    for op in ops:
        op_name = op.name.split('/')[-1]
        if op_name in ['Conv2D', 'MatMul', 'convolution']:
            assert(len(op.values()) == 1)
            print([x for x in op.inputs])
            output_size = np.prod(op.values()[0].shape, dtype=np.int32)
            print("{:.2f} MB".format(size_to_mb(output_size)))
            tot_size += output_size

    print("Total Size: {:.2f} MB".format(size_to_mb(tot_size)))


def test_forward_quant(sess, x, logits):
    preprocess = preprocessing_factory.get_preprocessing(args.model_name,
                                                         is_training=False)

    dataset_images, labels = imagenet.load_validation(
        args.input_dir, args.batch_size, preprocess=preprocess)

    num_batches = args.max_num_batches

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    res = Results()

    for i in range(num_batches):
        images, true_labels = sess.run([dataset_images, labels])

        res.start_timer()
        preds = sess.run(logits, feed_dict={x: images})
        res.end_timer()

        res.record_acc(preds, true_labels)
        res.print_results()
        sys.stdout.flush()

    coord.request_stop()
    coord.join(threads)


def main(_):

    with tf.Session() as sess:

        if args.model_name in ['vgg_16']:
            num_classes = 1000
        elif args.model_name in ['inception_v3']:
            num_classes = 1001
        else:
            raise AttributeError("unknown model {}".format(args.model_name))

        network_fn = nets_factory.get_network_fn(
            args.model_name, num_classes=num_classes, is_training=False
        )

        h, w = network_fn.default_image_size, network_fn.default_image_size
        images = tf.placeholder(dtype=tf.float32, shape=(args.batch_size, h, w, 3))
        labels = tf.placeholder(dtype=tf.float32, shape=(args.batch_size, num_classes))
        logits, _ = network_fn(images)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        

        if args.test_name == 'model_size':
            print_model_size(sess)
        elif args.test_name == 'forward':
            test_forward_quant(sess, images, logits)


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
