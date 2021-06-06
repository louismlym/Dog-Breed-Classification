import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.utils import save, restore, restore_latest


class TraceableSequential(nn.Sequential):
    def __init__(self, args, traceable=False):
        super(TraceableSequential, self).__init__(*args)
        self.traceable = traceable
        self.traces = []

    def forward(self, input_tensor: torch.Tensor):
        if self.traceable:
            self.traces = []
        for module in self._modules.values():
            module_input = None
            if self.traceable:
                module_input = input_tensor.detach()
            input_tensor = module(input_tensor)
            if self.traceable:
                module_output = input_tensor.detach()
                weight = None
                if hasattr(module, "weight") and len(module.weight.size()) == 4:
                    weight = module.weight.detach()
                self.traces.append((module_input, module_output, module, weight))
        return input_tensor

    def get_traces(self):
        return self.traces

    def set_traceable(self, traceable: bool = True):
        self.traceable = traceable

    def __str__(self):
        return super(TraceableSequential, self).__str__()


class BaseSavableNet(nn.Module):
    def __init__(self):
        super(BaseSavableNet, self).__init__()
        self.__best_accuracy_saved = None

    def classify(self, input_tensor: torch.Tensor):
        return F.softmax(self.forward(input_tensor), dim=1)

    def loss(self, prediction, label, reduction="mean"):
        loss_val = F.cross_entropy(prediction, label.squeeze(), reduction=reduction)
        return loss_val

    def save_model(self, file_path, num_to_keep=1):
        save(self, file_path, num_to_keep)

    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        if self.__best_accuracy_saved is None or self.__best_accuracy_saved < accuracy:
            self.__best_accuracy_saved = accuracy
            self.save_model(file_path, num_to_keep)

    def load_model(self, file_path):
        restore(self, file_path)

    def load_last_model(self, dir_path):
        return restore_latest(self, dir_path)


def make_traceable_net(net_builder, traceable, num_classes, pretrained):
    if not pretrained:
        if num_classes is None:
            raise Exception(
                "num_classes should not be None if not using pretrained network"
            )
        net = net_builder(num_classes=num_classes)
    else:
        net = net_builder(pretrained=pretrained)
    net.features = TraceableSequential(
        list(net.features.modules())[1:], traceable=traceable
    )
    return net


class TraceableAlexNet(BaseSavableNet):
    def __init__(self, num_classes=None, traceable=False, pretrained=False):
        super(TraceableAlexNet, self).__init__()
        self.traceable = traceable
        self.alexnet = make_traceable_net(
            models.alexnet,
            traceable=traceable,
            num_classes=num_classes,
            pretrained=pretrained,
        )

    def forward(self, input_tensor):
        return self.alexnet(input_tensor)

    def load_last_model(self, dir_path):
        return restore_latest(self, dir_path)

    def set_traceable(self, traceable=True):
        self.traceable = traceable
        self.alexnet.features.set_traceable(traceable)

    def get_traces(self):
        return self.alexnet.features.get_traces()


class TraceableVgg(BaseSavableNet):
    def __init__(self, num_classes=None, traceable=False, pretrained=False):
        super(TraceableVgg, self).__init__()
        self.traceable = traceable
        self.vgg = make_traceable_net(
            models.vgg19_bn,
            traceable=traceable,
            num_classes=num_classes,
            pretrained=pretrained,
        )

    def forward(self, input_tensor):
        return self.vgg(input_tensor)

    def set_traceable(self, traceable=True):
        self.traceable = traceable
        self.vgg.features.set_traceable(traceable)

    def get_traces(self):
        return self.vgg.features.get_traces()


class GenericTraceableNet(BaseSavableNet):
    def __init__(self, net: nn.Module, seq_attr: str, traceable: bool = False):
        super(GenericTraceableNet, self).__init__()
        if getattr(net, seq_attr).__class__.__name__ != "Sequential":
            raise Exception("the attribute seq_attr of net should be Sequential")

        traceable_seq = TraceableSequential(
            list(getattr(net, seq_attr).modules())[1:], traceable=traceable
        )
        setattr(net, seq_attr, traceable_seq)
        self.net = net
        self.traceable = traceable
        self.seq_attr = seq_attr

    def forward(self, input_tensor: torch.Tensor):
        return self.net(input_tensor)

    def set_traceable(self, traceable: bool = True):
        self.traceable = traceable
        getattr(self.net, self.seq_attr).set_traceable(traceable)

    def get_traces(self):
        return getattr(self.net, self.seq_attr).get_traces()


class TraceableCustomAlexNet(nn.Module):
    def __init__(self, num_classes, device, traceable=False):
        super(TraceableCustomAlexNet, self).__init__()
        pretrain_model = make_traceable_net(
            models.alexnet,
            traceable=traceable,
            pretrained=False,
            num_classes=1024
        )
        # for param in pretrain_model.parameters():
        #     # freeze parameters, for them to not be trainable
        #     param.requires_grad = False
        self.model = nn.Sequential(
            pretrain_model,
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
        self.traceable = traceable
        self.accuracy = None
        self.device = device

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return F.softmax(x, dim=1)
    
    def set_traceable(self, traceable=True):
        self.traceable = traceable
        self.pretrain_model.features.set_traceable(traceable)

    def get_traces(self):
        return self.pretrain_model.features.get_traces()

    def loss(self, prediction, label, reduction="mean"):
        loss_val = F.cross_entropy(prediction, label.squeeze(), reduction=reduction)
        return loss_val

    def save_model(self, file_path, num_to_keep=1):
        save(self, file_path, num_to_keep)

    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        if self.accuracy == None or accuracy > self.accuracy:
            self.accuracy = accuracy
            self.save_model(file_path, num_to_keep)

    def load_model(self, file_path):
        restore(self, file_path, self.device)

    def load_last_model(self, dir_path):
        return restore_latest(self, dir_path, self.device)

class TraceableCNN(nn.Module):
    def __init__(self, num_classes, device, traceable=False):
        super(TraceableCNN, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fcs = nn.Sequential(
            nn.Linear(25088, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.traceable = traceable
        self.accuracy = None
        self.device = device

    def forward(self, x: torch.Tensor):
        if self.traceable:
            self.traces = []
        for module in self.convs._modules.values():
            module_input = None
            if self.traceable:
                module_input = x.detach()
            x = module(x)
            if self.traceable:
                module_output = x.detach()
                weight = None
                if hasattr(module, "weight") and len(module.weight.size()) == 4:
                    weight = module.weight.detach()
                self.traces.append((module_input, module_output, module, weight))
        x = torch.flatten(x, 1)
        x = self.fcs(x)
        return F.softmax(x, dim=1)

    def loss(self, prediction, label, reduction="mean"):
        loss_val = F.cross_entropy(prediction, label.squeeze(), reduction=reduction)
        return loss_val

    def save_model(self, file_path, num_to_keep=1):
        save(self, file_path, num_to_keep)

    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        if self.accuracy == None or accuracy > self.accuracy:
            self.accuracy = accuracy
            self.save_model(file_path, num_to_keep)

    def load_model(self, file_path):
        restore(self, file_path, self.device)

    def load_last_model(self, dir_path):
        return restore_latest(self, dir_path, self.device)
