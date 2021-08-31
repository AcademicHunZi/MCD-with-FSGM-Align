import random

import torch

from utils import load_data, eval_pgd, eval_cw, eval_r_fgsm, test

g = torch.load("models_trained/syn2gtsrb_g.pth", map_location="cuda:3")
f1 = torch.load("models_trained/syn2gtsrb_c1.pth", map_location="cuda:3")
f2 = torch.load("models_trained/syn2gtsrb_c2.pth", map_location="cuda:3")
g.eval()
f1.eval()
f2.eval()
_, data = load_data("gtsrb", 256)

eps = 25 / 255
alpha = 0
acc1 = test(g, f1, data, device="cuda:3")
acc2 = test(g, f2, data, device="cuda:3")
print("normal_acc: acc1={}， acc2 = {}".format(acc1, acc2))

acc1 = eval_r_fgsm(g, f1, data, eps, alpha, device="cuda:3")
acc2 = eval_r_fgsm(g, f2, data, eps, alpha, device="cuda:3")
print("r_fgsm_acc: acc1={}， acc2 = {}".format(acc1, acc2))

acc1 = eval_cw(g, f1, data, device="cuda:3", c=2)
acc2 = eval_cw(g, f2, data, device="cuda:3", c=2)
print("cw_acc: acc1={}， acc2 = {}".format(acc1, acc2))

acc1 = eval_pgd(g, f1, data, bound=eps, attack_iter=30, step_size=2 / 255, device="cuda:3")
acc2 = eval_pgd(g, f2, data, bound=eps, attack_iter=30, step_size=2 / 255, device="cuda:3")
print("pgd_acc: acc1={}， acc2 = {}".format(acc1, acc2))
