import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
import ssl
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

BATCH_SIZE = 64
TEST_BATCH_SIZE = 1
VAL_PERCENTAGE = 0.1
CHECKPOINT_PATH = 'logs/exp/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ssl._create_default_https_context = ssl._create_unverified_context # to be able to download pretrained model

def get_data(train_path, test_path):
    transform_train = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomCrop(128, padding=8, padding_mode='edge'), # Take 128x128 crops from padded images
        transforms.RandomHorizontalFlip(),    # 50% of time flip image along y-axis
        transforms.RandomRotation(45),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
    classes = trainset.classes

    # sample some data from train set for validation set
    train_size = int(len(trainset) * (1 - VAL_PERCENTAGE))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
    valset.transform = transform_test
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)

    return {'train': trainloader, 'val': valloader, 'test': testloader, 'classes': classes}

def show_image(data):
    dataiter = iter(data['train'])
    images, labels = dataiter.next()
    images = images[:8]
    print(images.size())

    def imshow(img):
        # mpl.use('tkagg') # for Mac BigSur to not get segfault using plt.imshow
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig('images/train-data-samples.jpg')

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print("Labels:" + ', '.join('%9s' % data['classes'][j] for j in labels[:8]))

def train_epochs(net, data, epochs=1, start_epoch=0, lr=0.01, momentum=0.9, decay=0.0005, 
          verbose=1, print_every=10, state=None, checkpoint_path=None):
    net.to(device)
    net.train()
    train_losses, val_losses = ([], [])
    train_acc, val_acc = ([], [])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma = 0.3) # decays LR by gamma for every step_size

    # Load previous training state
    if state:
        net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch']
        train_losses, val_losses = (state['train_losses'], state['val_losses'])
        train_acc, val_acc = (state['train_acc'], state['val_acc'])
        lr = scheduler.get_last_lr()[0]
        print ("Learning rate: %f" % lr)
        for g in optimizer.param_groups:
            g['lr'] = scheduler.get_last_lr()[0]

    def train_or_eval(epoch, training=True):
        if training:
            net.train()
            dataloader = data['train']
        else:
            net.eval()
            dataloader = data['val']

        sum_loss = 0
        sum_correct = 0
        epoch_loss = 0

        for i, batch in enumerate(dataloader, 0):
            inputs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(training == True): # eval won't create any computational graph because it doesn't back propagate
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                sum_loss += loss.item()
                epoch_loss += loss.item() * inputs.size(0)

                if training == True:
                    loss.backward()  # autograd magic, computes all the partial derivatives
                    optimizer.step() # takes a step in gradient direction
                    train_losses.append(loss.item())
                else:
                    val_losses.append(loss.item())

                # calculate correct predictions
                predicted = outputs.max(1)[1]
                correct = torch.sum(predicted == labels.data)
                sum_correct += correct

                if i % print_every == print_every - 1:
                    if verbose:
                        if training == True:
                            print('Train ', end='')
                        else:
                            print('Val ', end='')
                        print('Epoch: {} Batch {} [{}/{} ({:.2f}%)] loss: {:.4f} accuracy: {:.6f}'.format(
                            epoch,
                            i + 1,
                            min((i + 1) * BATCH_SIZE, len(dataloader.dataset)),
                            len(dataloader.dataset),
                            100.0 * (i + 1) / len(dataloader),
                            sum_loss / print_every,
                            sum_correct / min((i + 1) * BATCH_SIZE, len(dataloader.dataset))
                            ))
                    sum_loss = 0.0
        avg_loss = epoch_loss / len(dataloader.dataset)
        accuracy = sum_correct / len(dataloader.dataset)
        if training == True:
            train_acc.append(accuracy)
        else:
            val_acc.append(accuracy)
        return avg_loss, accuracy, sum_correct

    best_epoch = None
    best_acc = 0

    for epoch in range(start_epoch, epochs):
        epoch_train_loss, epoch_train_acc, train_correct = train_or_eval(epoch, training=True)
        epoch_val_loss, epoch_val_acc, val_correct = train_or_eval(epoch, training=False)
        scheduler.step()

        print('=' * 50)
        print('Epoch {} TRAIN      - average loss: {:.4f} accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch, epoch_train_loss, train_correct, len(data['train'].dataset), epoch_train_acc * 100.0))
        print('Epoch {} VALIDATION - average loss: {:.4f} accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch, epoch_val_loss, val_correct, len(data['val'].dataset), epoch_val_acc * 100.0))
        print('=' * 50)

        if checkpoint_path:
            state = {'epoch': epoch+1, 'net': net.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'train_losses': train_losses,
                     'val_losses': val_losses, 'train_acc': train_acc, 'val_acc': val_acc}
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(state, checkpoint_path + 'checkpoint-%d.pkl'%(epoch+1))
        
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_epoch = epoch
    print('Best VALIDATION accuracy ({:.4f}%) at epoch {}'.format(best_acc, best_epoch))
    return train_losses, val_losses, train_acc, val_acc

def get_model(num_classes):
    resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
    resnet.fc = nn.Linear(2048, num_classes)
    return resnet

def get_state(chkpt):
    return torch.load(CHECKPOINT_PATH + chkpt)

def load_model(num_classes, chkpt, strict=True):
    model = get_model(num_classes)
    state = get_state(chkpt)
    model.load_state_dict(state['net'], strict=strict)
    return model

def train_model(train_path, test_path, exp_version, epochs, batch_size, lr, momentum, decay, print_every, verbose):
    CHECKPOINT_PATH = 'logs/' + exp_version + '/'
    BATCH_SIZE = batch_size
    print("Using device:", device)
    data = get_data(train_path, test_path)
    # show_image(data)
    resnet = get_model(len(data['classes']))
    print("Using mode:")
    print(resnet)
    train_epochs(resnet, data, epochs=epochs, lr=lr, momentum=momentum,
        decay=decay, print_every=print_every, checkpoint_path=CHECKPOINT_PATH,
        verbose=verbose)
