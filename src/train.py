import os
import torch
import multiprocessing
import time
import numpy as np
import pickle

from torchvision.transforms.transforms import Grayscale

from src import utils
from os import path
from src.model import TraceableAlexNet, TraceableCNN, TraceableCustomAlexNet
from torch import optim
from torchvision import datasets, transforms

# Define constants
TRAIN_PATH = "./data/train"
TEST_PATH = "./data/val"
EXPERIMENT_VERSION = "alexNet"  # change this to start a new experiment
LOG_PATH = "./logs/" + EXPERIMENT_VERSION + "/"
IMAGE_PATH = "./images"

# Define hyperparamters
BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9
USE_CUDA = True
PRINT_INTERVAL = 10
WEIGHT_DECAY = 0.0005

# Define Train and Test functions
def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    losses = []
    correct = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        # calculate correct predictions
        pred = output.max(1)[1]
        correct_mask = pred.eq(label.view_as(pred))
        num_correct = correct_mask.sum().item()
        correct += num_correct

        if batch_idx % log_interval == 0:
            if batch_idx == 0:
                acc = 0
            else:
                acc = correct / (batch_idx * len(data))
            print(
                "{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}".format(
                    time.ctime(time.time()),
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    acc
                )
            )
    train_loss = np.mean(losses)
    train_accuracy = 100.0 * correct / len(train_loader.dataset)

    print(
        "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            train_loss, correct, len(train_loader.dataset), train_accuracy
        )
    )

    return train_loss, train_accuracy


def test(model, device, test_loader, num_classes, log_interval=None):
    model.eval()
    test_loss = 0
    correct = 0
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss_on = model.loss(output, label, reduction="sum").item()
            test_loss += test_loss_on
            pred = output.max(1)[1]
            stacked = torch.stack((label, pred), dim=1)
            for i, p in enumerate(stacked):
                tl, pl = p.tolist()
                confusion_matrix[tl, pl] = confusion_matrix[tl, pl] + 1
            correct_mask = pred.eq(label.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct
            if log_interval is not None and batch_idx % log_interval == 0:
                print(
                    "{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        time.ctime(time.time()),
                        batch_idx * len(data),
                        len(test_loader.dataset),
                        100.0 * batch_idx / len(test_loader),
                        test_loss_on,
                    )
                )

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), test_accuracy
        )
    )
    return test_loss, test_accuracy, confusion_matrix


# Import and transform dataset (data augmentations)
transform_resize = transforms.Resize((256, 256))
transform_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) 
transform = transforms.Compose(
    [
        transform_resize,
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transform_normalize,
    ]
)
transform_test = transforms.Compose(
    [
        transform_resize,
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transform_normalize,
    ]
)


def train_model():
    data_train = datasets.ImageFolder(TRAIN_PATH, transform=transform)
    data_test = datasets.ImageFolder(TEST_PATH, transform=transform_test)

    NUM_CLASSES = len(data_train.classes)

    # Now the actual training code
    use_cuda, device = utils.get_device()
    print("Using device", device)
    print("num cpus:", multiprocessing.cpu_count())
    print("num classes:", NUM_CLASSES)

    kwargs = (
        {"num_workers": multiprocessing.cpu_count(), "pin_memory": True}
        if use_cuda
        else {}
    )

    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=BATCH_SIZE, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs
    )

    # - Example of how to save an image
    # images, labels = next(iter(train_loader))
    # utils.save_image("./images/example2-resize-horizontalFlip.png", images[0])

    # - Example of how to save image from dataloader
    # utils.save_image_from_dataloader(test_loader, 0, 72, "./images/correct-prediction-1", False)

    model = TraceableCustomAlexNet(num_classes=NUM_CLASSES, device=device).to(device)
    print(model)
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    start_epoch = model.load_last_model(LOG_PATH) + 1
    # read log
    if os.path.exists(LOG_PATH + "log.pkl"):
        train_losses, test_losses, test_accuracies, train_accuracies = pickle.load(
            open(LOG_PATH + "log.pkl", "rb")
        )
    else:
        train_losses, test_losses, test_accuracies, train_accuracies = ([], [], [], [])

    test_loss, test_accuracy, confusion_matrix = test(
        model, device, test_loader, NUM_CLASSES, log_interval=10
    )

    test_losses.append((start_epoch, test_loss))
    test_accuracies.append((start_epoch, test_accuracy))
    epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            train_loss, train_accuracy = train(
                model, device, train_loader, optimizer, epoch, PRINT_INTERVAL
            )
            test_loss, test_accuracy, confusion_matrix = test(
                model, device, test_loader, NUM_CLASSES
            )
            train_losses.append((epoch, train_loss))
            test_losses.append((epoch, test_loss))
            train_accuracies.append((epoch, train_accuracy))
            test_accuracies.append((epoch, test_accuracy))
            # write log
            if not os.path.exists(os.path.dirname(LOG_PATH + "log.pkl")):
                os.makedirs(os.path.dirname(LOG_PATH + "log.pkl"))
            pickle.dump(
                (train_losses, test_losses, test_accuracies),
                open(LOG_PATH + "log.pkl", "wb"),
            )

            model.save_best_model(test_accuracy, LOG_PATH + "%03d.pt" % epoch)

    except KeyboardInterrupt as ke:
        print("Interrupted")
    except:
        import traceback

        traceback.print_exc()
    finally:
        model.save_model(LOG_PATH + "%03d.pt" % epoch, 0)
        ep, val = zip(*train_losses)
        utils.save_plot(
            ep,
            val,
            "Train loss",
            "Epoch",
            "Error",
            path.join(IMAGE_PATH, "train-loss.jpg"),
        )
        ep, val = zip(*test_losses)
        utils.save_plot(
            ep,
            val,
            "Test loss",
            "Epoch",
            "Error",
            path.join(IMAGE_PATH, "test-loss.jpg"),
        )
        ep, val = zip(*train_accuracies)
        utils.save_plot(
            ep,
            val,
            "Train accuracy",
            "Epoch",
            "Accuracy (percentage)",
            path.join(IMAGE_PATH, "train-accuracy.jpg"),
        )
        ep, val = zip(*test_accuracies)
        utils.save_plot(
            ep,
            val,
            "Test accuracy",
            "Epoch",
            "Accuracy (percentage)",
            path.join(IMAGE_PATH, "test-accuracy.jpg"),
        )
        utils.save_confusion_matrix(
            confusion_matrix,
            data_train.classes,
            path.join(IMAGE_PATH, "confusion-matrix.jpg")
        )
