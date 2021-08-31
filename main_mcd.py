import argparse
import random
from multiprocessing.dummy import freeze_support

from MCD import MCD
import numpy as np
from utils import *
from torch import optim
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_name', default="mnist", type=str)
    parser.add_argument('--tgt_name', default="usps", type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8 / 255, type=float)
    parser.add_argument('--step_size', default=2 / 255, type=float)
    parser.add_argument('--bound', default=20 / 255, type=float)
    parser.add_argument('--attack_iter', default=30, type=int)
    parser.add_argument('--seed', default=1, type=int, )
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--grad_align_cos_lambda', default=0.05, type=int)
    parser.add_argument('--N', default=4, type=int)
    parser.add_argument('--log_interval', default=100, type=int)

    return parser.parse_args()


def main():
    args = get_args()
    if args.src_name == "syn":
        from model.syn2gtrsb import Feature, Predictor
    else:
        from model.svhn2mnist import Feature, Predictor

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    G = Feature().apply(init_weights).to(args.device)
    C1 = Predictor().apply(init_weights).to(args.device)
    C2 = Predictor().apply(init_weights).to(args.device)

    opt_g = optim.Adam(G.parameters(), lr=args.lr, weight_decay=5e-4)
    opt_c1 = optim.Adam(C1.parameters(), lr=args.lr, weight_decay=5e-4)
    opt_c2 = optim.Adam(C2.parameters(), lr=args.lr, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()

    dictionary = {'G': G, 'C1': C1, 'C2': C2, 'opt_g': opt_g, "opt_c1": opt_c1, 'opt_c2': opt_c2,
                  'criterion': criterion}

    mcd = MCD(dictionary, args)
    source_train, _ = load_data(args.src_name, args.batch_size)
    target_train, target_val = load_data(args.tgt_name, args.batch_size)

    mcd.train(source_train, target_train, target_val)


if __name__ == '__main__':
    freeze_support()
    main()
