import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import losses
class hard_region_Loss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, kernel_size=3):
        super(hard_region_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.kernel_size = kernel_size
        self.criterion = losses.softmax_mse_loss()  # 均方差损失函数
        
    def forward(self, predictions, labels, boundary_mask):
        

# 找到边界左右两边的像素
        left_boundary_pixels = predictions[:, :,  :-1,:,:][boundary_mask]
        right_boundary_pixels = predictions[:, :, 1:,:,: ][boundary_mask]

        # 找到边界左右两边的标签
        left_boundary_labels = labels[:, :-1,:,:][boundary_mask]
        right_boundary_labels = labels[:,1:,:,:][boundary_mask]
        #torch.Size([2, 112, 112, 80])
#########################
# 根据边界掩码和过渡区域掩码，提取边界像素和过渡区域像素
        boundary_pixels = predictions[boundary_mask]
        transition_pixels = predictions[boundary_mask]

        # 根据掩码提取相应的标签
        boundary_labels = labels[boundary_mask]
        transition_labels = labels[boundary_mask]

        # 使用局部卷积来捕获过渡信息
        trans_pixels = self.local_conv(transition_pixels, self.kernel_size)
        #right_boundary_pixels = self.local_conv(boundary_pixels, self.kernel_size)

        # 计算边界损失，考虑左右两边像素
        boundary_loss = self.criterion(boundary_pixels, boundary_labels)
        left_boundary_loss = self.criterion(left_boundary_pixels, left_boundary_labels)
        right_boundary_loss = self.criterion(right_boundary_pixels, right_boundary_labels)
###left_boundary_loss = self.criterion(left_boundary_pixels, left_boundary_labels)
  ##right_boundary_loss = self.criterion(right_boundary_pixels, right_boundary_labels)
        # 计算过渡区域损失
        transition_loss = self.criterion(trans_pixels, transition_labels)
        # 组合总损失
        total_loss = self.alpha * boundary_loss + self.beta * transition_loss + left_boundary_loss + right_boundary_loss

        return total_loss

    def local_conv(self, x, kernel_size):
        # 实现局部卷积操作来捕获周围像素的信息
        # 使用padding来确保输出尺寸与输入相同
        padding = (kernel_size - 1) // 2
        local_convolution = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        local_convolution.weight.data.fill_(1.0)  # 权重全为1，即平均卷积
        local_convolution.weight.requires_grad = False  # 权重不需要训练
        x = x.unsqueeze(1)  # 添加通道维度
        local_features = local_convolution(x)
        local_features = local_features.squeeze(1)  # 移除通道维度
        return local_features
def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent
def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    # input_softmax = F.softmax(input_logits, dim=1)
    # target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_logits-target_logits)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)
