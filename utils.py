import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

from attack import cw_l2_attack, pgd_attack, r_fgsm_attack


def get_input_grad(netG, netF, X, y, eps, delta_init='none', backprop=False):
    if delta_init == 'none':
        delta = torch.zeros_like(X, requires_grad=True)
    elif delta_init == 'random_uniform':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
    elif delta_init == 'random_corner':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
        delta = eps * torch.sign(delta)
    else:
        raise ValueError('wrong delta init')

    output = netF(netG(X + delta))
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
    if not backprop:
        grad, delta = grad.detach(), delta.detach()
    return grad


def get_uniform_delta(shape, eps, requires_grad=True):
    delta = torch.zeros(shape).to("cuda")
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta


def l2_norm_batch(v):
    norms = (v ** 2).sum([1, 2, 3]) ** 0.5
    return norms


def eval_pgd(netG, netF, data, bound, attack_iter, step_size, device):
    test_loss = 0
    test_acc = 0
    test_n = 0

    for img, label in data:
        img = img.to(device)
        label = label.to(device)

        img = pgd_attack(img, label, netG, netF, bound, attack_iter, step_size, device)
        pred = netF(netG(img))
        loss = F.cross_entropy(pred, label)
        test_loss += loss.item()
        test_acc += (pred.max(1)[1] == label).sum().item()
        test_n += label.size(0)
    return test_acc / test_n


def eval_cw(netG, netF, data, device, c=2):
    test_loss = 0
    test_acc = 0
    test_n = 0

    for img, label in data:
        img = img.to(device)
        label = label.to(device)

        img = cw_l2_attack(netG, netF, img, label, device, c)
        pred = netF(netG(img))
        loss = F.cross_entropy(pred, label)
        test_loss += loss.item()
        test_acc += (pred.max(1)[1] == label).sum().item()
        test_n += label.size(0)
    return test_acc / test_n


def eval_r_fgsm(netG, netF, data, eps, alpha, device):
    test_loss = 0
    test_acc = 0
    test_n = 0

    for img, label in data:
        img = img.to(device)
        label = label.to(device)

        img = r_fgsm_attack(img, label, netG, netF, eps, alpha, device)
        pred = netF(netG(img))
        loss = F.cross_entropy(pred, label)
        test_loss += loss.item()
        test_acc += (pred.max(1)[1] == label).sum().item()
        test_n += label.size(0)
    return test_acc / test_n


def load_img(path, transform, batch_size=128, shuffle=True):
    data_set = datasets.ImageFolder(root=path, transform=transform)
    data_set = torch.utils.data.DataLoader(data_set, shuffle=shuffle, batch_size=batch_size,
                                           num_workers=4, drop_last=True)
    return data_set


def test(netG, netF, data, device):
    test_loss = 0
    test_acc = 0
    test_n = 0

    for img, label in data:
        img = img.to(device)

        label = label.to(device)
        pred = netF(netG(img))
        loss = F.cross_entropy(pred, label)
        test_loss += loss.item()
        test_acc += (pred.max(1)[1] == label).sum().item()
        test_n += label.size(0)
    return test_acc / test_n


def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))


def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_normal(m.weight)
    if type(m) == nn.Linear:
        nn.init.xavier_normal(m.weight)


def load_data(data_name, batch_size):
    transform1 = transforms.Compose([
        transforms.Resize(40),
        transforms.CenterCrop(40),
        transforms.ToTensor(),
    ])

    transform2 = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    transform3 = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ])

    train_data, eval_data = None, None
    if data_name == "mnist":
        train_data = torchvision.datasets.MNIST(root="data", download=True, train=True, transform=transform2)
        eval_data = torchvision.datasets.MNIST(root="data", download=True, train=False, transform=transform2)
        train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        eval_data = DataLoader(eval_data, batch_size=batch_size, shuffle=True, num_workers=4)
    elif data_name == "svhn":
        train_data = torchvision.datasets.SVHN(root="data", download=True, split="train", transform=transform3)
        eval_data = torchvision.datasets.SVHN(root="data", download=True, split="test", transform=transform3)
        train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        eval_data = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=4)
    elif data_name == "usps":
        train_data = torchvision.datasets.USPS(root="data", download=True, train=True, transform=transform2)
        eval_data = torchvision.datasets.USPS(root="data", download=True, train=False, transform=transform2)
        train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        eval_data = DataLoader(eval_data, batch_size=batch_size, shuffle=True, num_workers=4)
    elif data_name == "syn":
        train_data = datasets.ImageFolder(root="../data/synthetic_data/train", transform=transform1)
        train_data = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=4,
                                                 drop_last=True)
    elif data_name == "gtsrb":
        train_data = datasets.ImageFolder(root="../data/GTSRB/Train", transform=transform1)
        train_data = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=4,
                                                 drop_last=True)

        eval_data = datasets.ImageFolder(root="../data/GTSRB/Test", transform=transform1)
        eval_data = torch.utils.data.DataLoader(eval_data, shuffle=True, batch_size=batch_size, num_workers=4,
                                                drop_last=True)
    else:
        ValueError("找不到数据集")

    return train_data, eval_data
