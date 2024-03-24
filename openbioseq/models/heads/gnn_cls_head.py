import torch
import torch.nn as nn
from torch import linalg
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, normal_init
from mmcv.runner import BaseModule

from ..utils import accuracy, accuracy_mixup, trunc_normal_init
from ..registry import HEADS
from ..builder import build_loss

class ArcFace(nn.Module):
    def __init__(self, cin, cout, s=8, m=0.5):
        super().__init__()
        self.s = s
        self.sin_m = torch.sin(torch.tensor(m))
        self.cos_m = torch.cos(torch.tensor(m))
        self.cout = cout
        self.fc = nn.Linear(cin, cout, bias=False)

    def forward(self, x, label=None):
        w_L2 = linalg.norm(self.fc.weight.detach(), dim=1, keepdim=True).T
        x_L2 = linalg.norm(x, dim=1, keepdim=True)
        cos = self.fc(x) / (x_L2 * w_L2)

        if label is not None:
            sin_m, cos_m = self.sin_m, self.cos_m
            one_hot = F.one_hot(label, num_classes=self.cout)
            sin = (1 - cos ** 2) ** 0.5
            angle_sum = cos * cos_m - sin * sin_m
            cos = angle_sum * one_hot + cos * (1 - one_hot)
            cos = cos * self.s
                        
        return cos

@HEADS.register_module
class GNN_ClsHead(BaseModule):
    """Simplest classifier head, with only one fc layer.
       *** Mixup and multi-label classification are supported ***
    
    Args:
        with_avg_pool (bool): Whether to use GAP before this head.
        loss (dict): Config of classification loss.
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of categories excluding the category.
        multi_label (bool): Whether to use one_hot like labels (requiring the
            multi-label classification loss). Notice that we support the
            single-label cls task to use the multi-label cls loss.
        frozen (bool): Whether to freeze the parameters.
    """

    def __init__(self,
                 with_avg_pool=False,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 in_channels=2048,
                 num_classes=1000,
                 multi_label=False,
                 frozen=False,
                 init_cfg=None):
        super(GNN_ClsHead, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.multi_label = multi_label

        # loss
        if loss is not None:
            assert isinstance(loss, dict)
            self.criterion = build_loss(loss)
        else:
            assert multi_label == False
            loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
            self.criterion = build_loss(loss)
        # fc layer
        # self.fc = nn.Linear(in_channels, num_classes)
        a = 1
        self.W_voice = nn.Parameter(torch.randn(in_channels // a, 3), requires_grad=True)
        self.W_moa = nn.Parameter(torch.randn(in_channels // a, 6), requires_grad=True)
        self.W_asp = nn.Parameter(torch.randn(in_channels // a, 3), requires_grad=True)
        self.W_poa = nn.Parameter(torch.randn(in_channels // a, 8), requires_grad=True)
        self.adj_matrx = nn.Parameter(torch.randn(20, 20), requires_grad=True)

        self.fc_moa = nn.Linear(in_channels, in_channels // a)
        self.fc_poa = nn.Linear(in_channels, in_channels // a)
        self.fc_voice = nn.Linear(in_channels, in_channels // a)
        self.fc_asp = nn.Linear(in_channels, in_channels // a)
        # self.fc_moa = nn.Linear(in_channels, in_channels)
        # self.fc_poa = nn.Linear(in_channels, in_channels)
        # self.fc_voice = nn.Linear(in_channels, in_channels)
        # self.fc_asp = nn.Linear(in_channels, in_channels)
        # self.fc1 = nn.Linear(20, in_channels)
        self.fc = nn.Linear(in_channels, num_classes)
        self.act = nn.ReLU()
        # self.fc = ArcFace(20, num_classes, m=0.5)
        if frozen:
            self.frozen()

    def frozen(self):
        self.fc.eval()
        for param in self.fc.parameters():
            param.requires_grad = False

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        super(GNN_ClsHead, self).init_weights()

        if self.init_cfg is not None:
            return
        assert init_linear in ['normal', 'kaiming', 'trunc_normal'], \
            "Undefined init_linear: {}".format(init_linear)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                elif init_linear == 'kaiming':
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif init_linear == 'trunc_normal':
                    trunc_normal_init(m, std=std, bias=bias)
            else:
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                elif init_linear == 'kaiming':
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif init_linear == 'trunc_normal':
                    trunc_normal_init(m, std=std, bias=bias)             

    def forward(self, x):
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            if x.dim() == 3:
                x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
            elif x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        else:
            x = x.reshape(x.size(0), -1)
        # get four representations
        merged_prototype = torch.cat((self.W_poa, self.W_moa, self.W_voice, self.W_asp), 1)
        # information aggregation
        merged_prototype = torch.matmul(merged_prototype, self.adj_matrx)
        feature_poa = self.act(self.fc_poa(x))
        feature_moa = self.act(self.fc_moa(x))
        feature_voice = self.act(self.fc_voice(x))
        feature_asp = self.act(self.fc_asp(x))
        x_voice = torch.matmul(feature_voice, merged_prototype[:, 14:17])
        x_poa = torch.matmul(feature_poa, merged_prototype[:, 0:8])
        x_moa = torch.matmul(feature_moa, merged_prototype[:, 8:14])
        x_asp = torch.matmul(feature_asp, merged_prototype[:, 17:20])
        # x = torch.cat((feature_poa, feature_moa, feature_voice, feature_asp), 1)
        # print(self.adj_matrx)
        # x_merged = self.act(self.fc1(x_merged))
        # x = self.fc(x)
        x = self.fc(x)
        # return [x]
        return [x_poa, x_moa, x_voice, x_asp, x]
        # return [x]

    def loss(self, cls_score, labels, **kwargs):
        """" cls loss forward
        
        Args:
            cls_score (list): Score should be [tensor].
            labels (tuple or tensor): Labels should be tensor [N, \*] by default.
                If labels as tuple, it's used for CE mixup, (gt_a, gt_b, lambda).
        """
        single_label = False
        losses = dict()
        # assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
        
        # computing loss
        if not isinstance(labels, tuple):
            # whether is the single label cls [N,] or multi-label cls [N,C]
            single_label = \
                labels.dim() == 1 or (labels.dim() == 2 and labels.shape[1] == 1)
            # Notice: we allow the single-label cls using multi-label loss, thus
            # * For single-label or multi-label cls, loss = loss.sum() / N
            avg_factor = labels.size(0)
            target = labels.clone()
            # print(labels.shape)
            if self.multi_label:
                # convert to onehot labels
                if single_label:
                    target = F.one_hot(target, num_classes=self.num_classes)
            # default onehot cls
            factor = 0.1
            # losses['loss'] = self.criterion(
            #     cls_score[-1], target[:, -1], avg_factor=avg_factor, **kwargs)
            # target[:, 0][torch.randperm(target.size()[0])]
            # losses['loss'] = self.criterion(
            #     cls_score[-1], target[:, -1], avg_factor=avg_factor, **kwargs) + factor * self.criterion(
            #     cls_score[0], target[:, 0], avg_factor=avg_factor, **kwargs) + factor * self.criterion(
            #     cls_score[1], target[:, 1], avg_factor=avg_factor, **kwargs) + factor * self.criterion(
            #     cls_score[2], target[:, 2], avg_factor=avg_factor, **kwargs) + factor * self.criterion(
            #     cls_score[3], target[:, 3], avg_factor=avg_factor, **kwargs)
            losses['loss'] = self.criterion(
                cls_score[-1], target[:, -1], avg_factor=avg_factor, **kwargs) + factor * self.criterion(
                cls_score[0], target[:, 0][torch.randperm(target.size()[0])], avg_factor=avg_factor, **kwargs) + factor * self.criterion(
                cls_score[1], target[:, 1][torch.randperm(target.size()[0])], avg_factor=avg_factor, **kwargs) + factor * self.criterion(
                cls_score[2], target[:, 2][torch.randperm(target.size()[0])], avg_factor=avg_factor, **kwargs) + factor * self.criterion(
                cls_score[3], target[:, 3][torch.randperm(target.size()[0])], avg_factor=avg_factor, **kwargs)
            # # compute accuracy
            # losses['acc'] = accuracy(cls_score[-1], labels[:, -1])
            losses['acc'] = accuracy(cls_score[-1], labels[:, -1])
        else:
            # mixup classification
            y_a, y_b, lam = labels
            if isinstance(lam, torch.Tensor):  # lam is scalar or tensor [N,1]
                lam = lam.unsqueeze(-1)
            # whether is the single label cls [N,] or multi-label cls [N,C]
            single_label = \
                y_a.dim() == 1 or (y_a.dim() == 2 and y_a.shape[1] == 1)
            # Notice: we allow the single-label cls using multi-label loss, thus
            # * For single-label or multi-label cls, loss = loss.sum() / N
            avg_factor = y_a.size(0)

            if not self.multi_label:
                losses['loss'] = \
                    self.criterion(cls_score[0], y_a, avg_factor=avg_factor, **kwargs) * lam + \
                    self.criterion(cls_score[0], y_b, avg_factor=avg_factor, **kwargs) * (1 - lam)
            else:
                # convert to onehot labels
                if single_label:
                    y_a = F.one_hot(y_a, num_classes=self.num_classes)
                    y_b = F.one_hot(y_b, num_classes=self.num_classes)
                # mixup onehot like labels, using a multi-label loss
                y_mixed = lam * y_a + (1 - lam) * y_b
                losses['loss'] = self.criterion(
                    cls_score[0], y_mixed, avg_factor=avg_factor, **kwargs)
            # compute accuracy
            losses['acc'] = accuracy(cls_score[0], labels[0])
            losses['acc_mix'] = accuracy_mixup(cls_score[0], labels)
        return losses
