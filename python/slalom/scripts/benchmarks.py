"""Run Benchmarks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from python.slalom.sgxdnn import SGXDNNUtils


def main():
    sgxutils = SGXDNNUtils(args.use_sgx)
    sgxutils.benchmark(args.threads)
    sgxutils.destroy()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_sgx', action='store_true')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    main()
