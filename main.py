import sys

from src.train import train_model

def train(argv):
  len_argv = len(argv)
  if len_argv > 2 or (len_argv == 2 and argv[1] != "--train"):
    print(f"Usage: {argv[0]} [--train]")
    sys.exit(2)
  return len_argv == 2


def main(argv):
  if train(argv):
    train_model()


if __name__ == "__main__":
  main(sys.argv)