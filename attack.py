import torchvision
from torch import nn, optim
import torch
import torch.nn.functional as F


def cw_l2_attack(netG, netF, images, labels, device, targeted=False, c=1, kappa=0, max_iter=1000,
                 learning_rate=0.01):
    images = images.to(device)
    labels = labels.to(device)

    # Define f-function
    def f(x):

        outputs = netF(netG(x))
        one_hot_label = F.one_hot(labels)
        i, _ = torch.max((1 - one_hot_label) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_label.bool())

        # If targeted, optimize for making the other class most likely
        if targeted:
            return torch.clamp(i - j, min=-kappa)

        # If untargeted, optimize for making the other class most likely
        else:
            return torch.clamp(j - i, min=-kappa)

    w = torch.zeros_like(images, requires_grad=True).to(device)

    optimizer = optim.Adam([w], lr=learning_rate)

    for step in range(max_iter):
        a = 1 / 2 * (nn.Tanh()(w) + 1)
        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c * f(a))
        cost = loss1 + loss2
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    attack_images = 1 / 2 * (nn.Tanh()(w) + 1)
    # torchvision.utils.save_image(attack_images, "result/cw.png")
    return attack_images


def pgd_attack(img, label, netG, netF, eps, itr_pgd_attack, step_size, device):
    delta = torch.FloatTensor(img.shape[0], img.shape[1],
                              img.shape[2], img.shape[3]).uniform_(-eps, eps).to(device)
    delta = torch.clamp(delta + img, 0, 1) - img
    delta = delta.to(device)
    delta.requires_grad = True
    for i in range(itr_pgd_attack):
        # 清空梯度
        netG.zero_grad()
        netF.zero_grad()
        pred = netF(netG(img + delta))
        loss = F.cross_entropy(pred, label)
        loss.backward()
        grad = delta.grad.detach()
        grad = grad.sign()
        delta.data = delta + step_size * grad
        delta.data = torch.clamp(delta, -eps, eps)
        delta.data = torch.clamp(delta + img, 0, 1) - img
    # torchvision.utils.save_image(img + delta, "result/pgd.png")
    return img + delta


def r_fgsm_attack(images, labels, netG, netF, eps, alpha, device):
    images = images.to(device)
    labels = labels.to(device)

    images_new = images + alpha * torch.randn_like(images).sign()
    images_new.requires_grad = True
    outputs = netF(netG(images_new))

    netF.zero_grad()
    netG.zero_grad()

    cost = F.cross_entropy(outputs, labels)
    cost.backward()

    attack_images = images_new + (eps - alpha) * images_new.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    # torchvision.utils.save_image(attack_images, "result/rfgsm.png")
    return attack_images
