import os
from argparse import ArgumentParser

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
    print(f'Dataset: {args.dataset}')
    print(f'Acc: {acc}')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
