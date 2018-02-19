from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from timeit import default_timer as timer


class Results(object):

    def __init__(self, k=5):
        self.top1_acc = 0.0
        self.topk_acc = 0.0
        self.tot_times = []
        self.start = 0
        self.pred_counts = []
        self.k = k

    def start_timer(self):
        self.start = timer()

    def end_timer(self):
        self.tot_times.append(timer() - self.start)

    def record_acc(self, preds, true_labels):
        top1, topk = get_topk_acc(preds, true_labels, k=self.k)
        self.top1_acc += top1
        self.topk_acc += topk
        self.pred_counts.append(len(preds))

    def print_results(self):
        print("\ttop1 err: {:.1f}".format(100 * (1.0 - self.top1_acc / np.sum(self.pred_counts))))
        print("\ttop{} err: {:.1f}".format(self.k, 100 * (1.0 - self.topk_acc / np.sum(self.pred_counts))))
        if len(self.tot_times) > 1:
            print("\tprocess one image per {:.3f} s".format(np.sum(self.tot_times[1:]) / (1.0*np.sum(self.pred_counts[1:]))))
        else:
            print("\tprocess one image per {:.3f} s".format(np.sum(self.tot_times) / (1.0*np.sum(self.pred_counts))))


def get_topk_acc(preds, true_labels, k=5):
    preds = preds.argsort(axis=1)[:, -k:][:, ::-1] + 1
    return np.sum(preds[:, 0] == true_labels), \
           np.sum(np.any(preds.T == true_labels, axis=0))
