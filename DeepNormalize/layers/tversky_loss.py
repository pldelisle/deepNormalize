import torch
import torch.nn as nn
from torch.autograd import Function


class TverskyLoss(nn.Module):

    def __init__(self, alpha, beta, eps, n_classes):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.n_classes = n_classes

    def forward(self, y_pred, y_true):
        one_hot = torch.tensor((), dtype=torch.float)
        one_hot = one_hot.new_zeros(y_pred.shape, device="cuda")
        one_hot.scatter_(1, y_true.long(), 1)
        one_hot.cuda()

        ones = torch.tensor((), dtype=torch.float)
        ones = ones.new_ones(y_pred.shape, device="cuda")

        # self.save_for_backward(y_pred, one_hot)

        self.p0 = y_true
        self.p1 = ones - y_pred
        self.g0 = one_hot
        self.g1 = ones - one_hot

        tp = torch.sum(self.p0 * self.g0)
        fp = self.alpha * torch.sum(self.p0 * self.g1)
        fn = self.beta * torch.sum(self.p1 * self.g0)

        numerator = tp
        denominator = tp + fp + fn + self.eps
        score = torch.div(numerator, denominator)
        return 1.0 - torch.sum(score)

    # def backward(self, grad_output):
    #
    #     input, target = self.saved_variables
    #     grad_input = grad_target = None
    #
    #     if self.needs_input_grad[0]:
    #         grad_input = grad_output * self.beta * self.g1 * torch.sum(self.p0 * self.g0) \
    #                      / torch.pow((torch.sum(self.p0 * self.g0) + self.alpha * torch.sum(self.p0 * self.g1) + self.beta * (self.p1 * self.g0)), 2)
    #     if self.needs_input_grad[1]:
    #         grad_target = None
    #
    #     return grad_input, grad_target


class DiceLoss(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

    def dice_coeff(input, target):
        """Dice coeff for batches"""
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        for i, c in enumerate(zip(input, target)):
            s = s + DiceLoss().forward(c[0], c[1])

        return s / (i + 1)
