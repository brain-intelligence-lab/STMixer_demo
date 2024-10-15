import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, others) = ctx.saved_tensors
        gama = others[0].item()

        grad_input = grad_output.clone()
        tmp = ((1 / gama) * (1 / gama) * (gama - input.abs())).clamp(min=0)
        grad_input = grad_input * tmp
        return grad_input, None


class DSPIKE(nn.Module):
    def __init__(self, region=1.0):
        super(DSPIKE, self).__init__()
        self.region = region

    def forward(self, x, temp):
        out_bpn = torch.clamp(x, -self.region, self.region)
        out_bp = (torch.tanh(temp * out_bpn)) / \
                 (2 * np.tanh(self.region * temp)) + 0.5
        out_s = (x >= 0).float()
        return (out_s.float() - out_bp).detach() + out_bp

class ZDSPIKE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, others) = ctx.saved_tensors
        gama = others[0].item()
        region = 1.0
        k = 1 / (2 * np.tanh(gama * region))
        grad_input = grad_output.clone()
        m = torch.tanh(gama * input)
        tmp = k * (1 - m * m) * gama
        tmp[input > region] = 0
        tmp[input < -region] = 0
        grad_input = grad_input * tmp
        return grad_input, None

class LIFSpike(nn.Module):
    def __init__(self, T=1, thresh=1.0, tau=0.5, gamma=2.5, use_ann=False):
        super(LIFSpike, self).__init__()
        self.use_ann = use_ann
        self.act_ann = nn.ReLU()
        self.snn_act = ZDSPIKE.apply
        #self.snn_act = ZIF.apply
        self.T = T # time steps
        self.thresh = thresh
        self.tau = tau
        self.gamma = gamma
        self.mem_detach = False
        self.soft_reset = True

    def forward(self, x):
        if self.use_ann:
            return self.act_ann(x)
        else:
            if self.T == 1:
                spike =  self.snn_act(x - self.thresh, self.gamma)
                # TODO: maybe compute the KL divergence between the two distributions
                return spike
            else:
                # adjust the size of x (B*T,C,H,W) to (B,T,C,H,W)
                x = x.view(self.T, -1, *x.shape[1:])
                mem = torch.zeros_like(x[0])
                spikes = torch.zeros_like(x)
                for t in range(self.T):
                    # calculate the membrane potential
                    mem = mem * self.tau + x[t,...]
                    spike = self.snn_act(mem - self.thresh, self.gamma)
                    spikes[t,...] = spike
                    if self.soft_reset:
                        mem = mem - self.thresh * spike
                    else:
                        if self.mem_detach:
                            mem = mem * (1 - spike.detach())
                        else:
                            mem = mem * (1 - spike)
                # adjust the size of spikes (B,T,C,H,W) to (B*T,C,H,W)
                return spikes.view(-1, *spikes.shape[2:])

class ExpandTime(nn.Module):
    def __init__(self, T=1):
        super(ExpandTime, self).__init__()
        self.T = T

    def forward(self, x):
        x_seq = x[None,:,:,:,:]
        x_seq = x_seq.repeat(self.T, 1 , 1, 1, 1)
        # adjust the size of spikes (B,T,C,H,W) to (B*T,C,H,W)
        return x_seq.view(-1, *x_seq.shape[2:])

class RateEncoding(nn.Module):
    def __init__(self, T=1):
        super(RateEncoding, self).__init__()
        self.T = T

    def forward(self, x):
        x_seq = x[None, :, :, :, :]
        x_seq = x_seq.repeat(self.T, 1, 1, 1, 1)
        x_seq = x_seq.view(-1, *x_seq.shape[2:])
        # poision noise
        noise = torch.randn_like(x_seq)
        x_seq = (x_seq>=noise).float().detach()
        return x_seq

if __name__ == '__main__':
    x = torch.randn(2,3,8,8)
    expt = ExpandTime(T=2)
    y = expt(x)
    print(y.shape)
