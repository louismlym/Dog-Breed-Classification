import sys

from src.train import train_model

TRAIN_PATH = 'data/train'
TEST_PATH = 'data/testing'
EXP_VERSION = 'temp'
MODEL = 'resnext50_32x4d'
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9

def usage(argv):
  print(f"Usage: {argv[0]} [--train] [<TRAIN_PATH>] [<TEST_PATH>] [<EXP_VERSION>]")
  sys.exit(2)

def train(argv):
  len_argv = len(argv)
  if (len_argv == 2 or len_argv == 5) and argv[1] != "--train":
    usage(argv)
  return len_argv == 2 or len_argv == 5


def main(argv):
  if train(argv):
    train_path = "./data/train"
    test_path = "./data/val"
    exp_version = "resnext"
    if len(argv) == 5:
      train_path = argv[2]
      test_path = argv[3]
      exp_version = argv[4]
    train_model(train_path, test_path, exp_version)


if __name__ == "__main__":
  # main(sys.argv)
  train_model(TRAIN_PATH, TEST_PATH, EXP_VERSION, MODEL)