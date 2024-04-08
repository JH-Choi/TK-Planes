import torch


class LimitGradLayer(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, mask):
        ctx.mask = mask #save_for_backward(mask)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.mask #saved_tensors
        return grad_output * mask, None
        #limit = 1e-5
        #return torch.clip(grad_output,-limit,limit)