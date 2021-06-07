import sys
import argparse

from src.train import train_model

TRAIN_PATH = 'data/train'
TEST_PATH = 'data/testing'
EXP_VERSION = 'exp'
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
DECAY = 0.0005
PRINT_EVERY = 10
NUM_EPOCHS = 1

def main():
    parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--train', '-t', action='store_true',
                        help="flag to train the model")
    parser.add_argument('--train-path', nargs=None, default=TRAIN_PATH,
                        help="define path for training dataset")
    parser.add_argument('--test-path', nargs=None, default=TEST_PATH,
                        help="define path for test dataset")
    parser.add_argument('--exp-version', '-exp', nargs=None, default=EXP_VERSION,
                        help="define experiment version")
    parser.add_argument('--epochs', '-e', type=int, nargs=None, default=NUM_EPOCHS,
                        help="define number of epochs to train the model")
    parser.add_argument('--batch-size', '-bs', type=int, nargs=None, default=BATCH_SIZE,
                        help="define batch size")
    parser.add_argument('--learning-rate', '-lr', type=float, nargs=None, default=LEARNING_RATE,
                        help="define learning rate")
    parser.add_argument('--momentum', '-m', type=float, nargs=None, default=MOMENTUM,
                        help="define momentum for optimizer")
    parser.add_argument('--weight-decay', '-wd', type=float, nargs=None, default=DECAY,
                        help="define weight decay for optimizer")
    parser.add_argument('--print-every', '-p', type=int, nargs=None, default=PRINT_EVERY,
                        help="define print for every (batch)")
    parser.add_argument('--no-verbose', action='store_true',
                        help="flag to not print during training")

    args = parser.parse_args()
    if args.train == True:
        print('Params: ', args)
        train_model(
            train_path=args.train_path,
            test_path=args.test_path,
            exp_version=args.exp_version,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            momentum=args.momentum,
            decay=args.weight_decay,
            print_every=args.print_every,
            verbose=(not args.no_verbose))
    else:
        parser.error("must flag --train or -t to train the model")

if __name__ == "__main__":
    main()