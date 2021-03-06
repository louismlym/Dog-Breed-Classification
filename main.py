import sys
import argparse
from src.save_figs import save_figs

from src.train import train_model, write_submission_file

TRAIN_PATH = 'data/train'
TEST_PATH = 'data/testing'
FIGURES_PATH = 'figures'
EXP_VERSION = 'exp'
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
DECAY = 0.0005
PRINT_EVERY = 10
NUM_EPOCHS = 1
STEP_SIZE = 5
GAMMA = 0.3
CHECKPOINT = 'checkpoint-1.pkl'

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
    parser.add_argument('--step-size', '-ss', type=int, nargs=None, default=STEP_SIZE,
                        help="define step size for LR scheduler")
    parser.add_argument('--gamma', '-g', type=float, nargs=None, default=GAMMA,
                        help="define gamma for LR scheduler")
    parser.add_argument('--submission', '-sub', action='store_true',
                        help="flag to build submission csv file")
    parser.add_argument('--checkpoint', '-chkpt', nargs=None, default=CHECKPOINT,
                        help="define path for test dataset")
    parser.add_argument('--save-figs', action='store_true',
                        help="flag to save figs from a checkpoint")

    args = parser.parse_args()
    if args.train == True:
        print('Params:', args)
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
            verbose=(not args.no_verbose),
            step_size=args.step_size,
            gamma=args.gamma)
    elif args.submission == True:
        print('Params:', args)
        write_submission_file(
            train_path=args.train_path,
            test_path=args.test_path,
            exp_version=args.exp_version,
            checkpoint_name=args.checkpoint
        )
    elif args.save_figs == True:
        print('Params:', args)
        save_figs(
            exp_version=args.exp_version,
            checkpoint_name=args.checkpoint,
            figures_path=FIGURES_PATH
        )
    else:
        parser.error("must flag --train or -t to train the model or --submission to create csv file")

if __name__ == "__main__":
    main()