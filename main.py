import sys

from src.train import train_model

def usage():
  print(f"Usage: {argv[0]} [--train] [<TRAIN_PATH>] [<TEST_PATH>]")
  sys.exit(2)

def train(argv):
  len_argv = len(argv)
  if (len_argv == 2 or len_argv == 4) and argv[1] != "--train":
    usage()
  return len_argv == 2 or len_argv == 4


def main(argv):
  if train(argv):
    train_path = "./data/train"
    test_path = "./data/val"
    if len(argv) == 4:
      train_path = argv[2]
      test_path = argv[3]
    train_model(train_path, test_path)


if __name__ == "__main__":
  main(sys.argv)