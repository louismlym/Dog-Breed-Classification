import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import re
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def smooth(x, size):
  return np.convolve(x, np.ones(size)/size, mode='valid')

def save_plot(x_values, title, xlabel, ylabel, filename, toSmooth=False, epochs=None):
    """Plots a line graph

    Args:
        x_values(list or np.array): x values for the line
        title(str): Title for the plot
        xlabel(str): Label for the x axis
        ylabel(str): label for the y axis
    """


    plt.figure(figsize=(12, 7))
    if epochs is not None:
        batch_per_epoch = int(len(x_values) / epochs)
        plt.xticks([batch_per_epoch * e for e in range(epochs + 1)])
    else:
        plt.xticks([e for e in range(len(x_values))])
    if toSmooth:
        x_values = smooth(x_values, 20)
    plt.plot(x_values)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)

    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(filename)
    print('Saving figure in ' + filename)

def save_figs(exp_version, checkpoint_name, figures_path):
    checkpoint_path = 'logs/' + exp_version + '/'
    state = torch.load(checkpoint_path + checkpoint_name, map_location=device)
    train_losses = state['train_losses']
    val_losses = state['val_losses']
    if isinstance(state['train_acc'][0], int):
        train_acc = state['train_acc']
        val_acc = state['val_acc']
    else:
        train_acc = [acc.item() for acc in state['train_acc']]
        val_acc = [acc.item() for acc in state['val_acc']]
    num_epochs = len(train_acc)

    save_plot(
        val_losses,
        'Validation Losses Per Batch Over {} Epochs'.format(num_epochs),
        'Batch\nNote: {} batches contribute 1 epoch'.format(int(len(val_losses) / num_epochs)),
        'Loss',
        os.path.join(figures_path, exp_version + '/val-losses.jpg'),
        toSmooth=True,
        epochs=num_epochs)

    save_plot(
        train_losses,
        'Train Losses Per Batch Over {} Epochs'.format(num_epochs),
        'Batch\nNote: {} batches contribute 1 epoch'.format(int(len(train_losses) / num_epochs)),
        'Loss',
        os.path.join(figures_path, exp_version + '/train-losses.jpg'),
        toSmooth=True,
        epochs=num_epochs)

    save_plot(
        val_acc,
        'Validation Accuracy Over {} Epochs'.format(num_epochs),
        'Epoch',
        'Accuracy',
        os.path.join(figures_path, exp_version + '/val-acc.jpg'))

    save_plot(
        train_acc,
        'Train Accuracy Over {} Epochs'.format(num_epochs),
        'Epoch',
        'Accuracy',
        os.path.join(figures_path, exp_version + '/train-acc.jpg'))