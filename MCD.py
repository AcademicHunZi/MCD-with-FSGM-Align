import logging
import random
import os
import torch
import torch.nn.functional as F
from attack import cw_l2_attack
from utils import get_input_grad, l2_norm_batch, discrepancy, test, eval_pgd, eval_cw, eval_r_fgsm


class MCD:
    def __init__(self, dictionary, args):
        super(MCD, self).__init__()
        self.G = dictionary['G']
        self.C1 = dictionary['C1']
        self.C2 = dictionary['C2']
        self.opt_g = dictionary['opt_g']
        self.opt_c1 = dictionary['opt_c1']
        self.opt_c2 = dictionary['opt_c2']
        self.criterion = dictionary['criterion']
        self.device = args.device
        self.epochs = args.epochs
        self.epsilon = args.epsilon
        self.grad_align_cos_lambda = args.grad_align_cos_lambda
        self.N = args.N
        self.log_interval = args.log_interval
        self.attack_iter = args.attack_iter
        self.step_size = args.step_size
        self.bound = args.bound

        self.mix_data = int(args.batch_size / 8)

    def train(self, src_data, tgt_data, eval_data):
        self.train_mode()

        for epoch in range(self.epochs):

            for batch_idx, ((img_s, labels), (img_t, _)) in enumerate(zip(src_data, tgt_data)):
                img_s = img_s.to(self.device)
                labels = labels.to(self.device)
                img_t = img_t.to(self.device)
                '''
                step 1
                '''
                # 生成对抗扰动
                self.zero_grad()
                reg, delta = self.get_reg_loss_delta(img_s, labels)
                self.zero_grad()
                # 源域的数据，必须添加对抗扰动
                feature_s = self.G(img_s + delta)
                out_s1 = self.C1(feature_s)
                out_s2 = self.C2(feature_s)

                loss_s1 = self.criterion(out_s1, labels)
                loss_s2 = self.criterion(out_s2, labels)
                loss_s = loss_s1 + loss_s2 + reg

                loss_s.backward()
                self.opt_g.step()
                self.opt_c1.step()
                self.opt_c2.step()
                '''
                step 2
                '''
                # 生成对抗扰动

                self.zero_grad()
                reg, delta = self.get_reg_loss_delta(img_s, labels)
                self.zero_grad()

                feature_s = self.G(img_s + delta)
                out_s1 = self.C1(feature_s)
                out_s2 = self.C2(feature_s)

                index = random.sample(range(0, img_s.shape[0]), self.mix_data)
                feature_t = self.G(img_t[index] + delta[index])

                out_t1 = self.C1(feature_t)
                out_t2 = self.C2(feature_t)
                loss_s1 = self.criterion(out_s1, labels)
                loss_s2 = self.criterion(out_s2, labels)
                loss_s = loss_s1 + loss_s2 + reg
                loss_discrepancy = discrepancy(out_t1, out_t2)
                loss = loss_s - loss_discrepancy
                loss.backward()
                self.opt_c1.step()
                self.opt_c2.step()
                self.zero_grad()

                '''
                step 3
                '''
                for i in range(self.N):
                    self.zero_grad()
                    index = random.sample(range(0, img_s.shape[0]), self.mix_data)
                    reg, delta = self.get_reg_loss_delta(img_s[index], labels[index])
                    img_t[index] += delta
                    feature_t = self.G(img_t)
                    out_t1 = self.C1(feature_t)
                    out_t2 = self.C2(feature_t)
                    loss_discrepancy = discrepancy(out_t1, out_t2)
                    loss_discrepancy.backward()
                    self.opt_g.step()
                    self.zero_grad()

            if (epoch + 1) % 1 == 0:
                self.eval_mode()
                test_acc1 = test(self.G, self.C1, eval_data, self.device)
                test_acc2 = test(self.G, self.C2, eval_data, self.device)
                pgd_acc1 = eval_pgd(self.G, self.C1, eval_data, bound=self.bound, attack_iter=30, step_size=2 / 255,
                                    device=self.device)
                pgd_acc2 = eval_pgd(self.G, self.C2, eval_data, bound=self.bound, attack_iter=30, step_size=2 / 255,
                                    device=self.device)
                print(
                    "normal_acc: acc1={}， acc2 = {}, pgd_acc: acc1={}， acc2 = {}".format(test_acc1, test_acc2, pgd_acc1,
                                                                                         pgd_acc2))
                self.train_mode()

    def get_reg_loss_delta(self, img_s, labels):
        # 生成对抗扰动
        delta = torch.zeros_like(img_s).to(self.device)
        delta.requires_grad = True
        reg = torch.zeros(1).to(self.device)[0]
        order = random.randint(0, 1)
        if order == 0:
            pred = self.C1(self.G(img_s + delta))
        else:
            pred = self.C2(self.G(img_s + delta))

        loss = self.criterion(pred, labels)
        loss.backward()
        grad = delta.grad.detach()
        delta = self.epsilon * torch.sign(grad)
        delta = torch.clamp(delta + img_s, 0, 1) - img_s
        delta = delta.detach()
        self.zero_grad()
        # 梯度对齐损失
        if self.grad_align_cos_lambda != 0.0 and order == 0:
            reg += self.get_regLoss(self.G, self.C1, img_s, labels, grad)
        if self.grad_align_cos_lambda != 0.0 and order == 1:
            reg += self.get_regLoss(self.G, self.C2, img_s, labels, grad)
        self.zero_grad()

        return reg, delta

    def get_regLoss(self, netG, netF, img, label, grad1):
        grad2 = get_input_grad(netG, netF, img, label, self.epsilon, delta_init='random_uniform', backprop=True)
        grad1, grad2 = grad1.reshape(len(grad1), -1), grad2.reshape(len(grad2), -1)
        cos = F.cosine_similarity(grad1, grad2, 1)
        reg = self.grad_align_cos_lambda * (1.0 - cos.mean())
        return reg

    def zero_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()

    def train_mode(self):
        self.G.train()
        self.C1.train()
        self.C2.train()

    def eval_mode(self):
        self.G.eval()
        self.C1.eval()
        self.C2.eval()
