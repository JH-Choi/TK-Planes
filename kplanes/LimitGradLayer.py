import torch


class LimitGradLayer(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, mask):
        ctx.save_for_backward(mask)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask, None
