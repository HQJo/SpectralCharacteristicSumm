import os
from argparse import ArgumentParser

import pandas as pd

import utils
import evaluation

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='Dataset name')


def main(args):
    graphs, labels = utils.load_dataset(args.dataset)
    acc = evaluation.eval_NetLSD(graphs, labels)
    print('Accuracy:', acc)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
