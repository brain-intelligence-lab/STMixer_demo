import torch
import torch.nn as nn
import torch.nn.functional as F

def SoftCrossEntropy(inputs, target, temperature):
    log_likelihood = -F.log_softmax(inputs / temperature, dim=1)
    batch = inputs.shape[0]
    loss = torch.sum(torch.mul(log_likelihood, F.softmax(target.detach() / temperature, dim=1))) / batch
    return loss

def simple_loss(p, z):
    p = nn.functional.normalize(p, dim=1)
    z = nn.functional.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()

def SoftKL(inputs, target, temperature):
    criterion = nn.KLDivLoss(reduction = 'batchmean')
    log_likelihood = F.log_softmax(inputs / temperature, dim=1)
    # batch = inputs.shape[0]
    target_pro = F.softmax(target.detach() / temperature, dim=1)
    loss = criterion(log_likelihood, target_pro)
    return loss

class SBDistillationLoss(nn.Module):
    def __init__(self, criterion, temperature=3.0, alpha=1.0, lamb=0.333):
        super(SBDistillationLoss, self).__init__()
        self.criterion = criterion
        self.temperature = temperature
        self.lamb = lamb  # the weight of the soft loss
        self.alpha = alpha  # the weight of the auxiliary output hard loss
        self.special_loss = criterion# nn.CrossEntropyLoss(label_smoothing=0.2)

    def forward(self, inputs, target):
        # inputs is a list of network output [final output, intermediate outputs]
        finput = inputs[0]
        hard_loss = self.criterion(finput, target)
        soft_loss = 0  # soft loss is the kl divergence loss to make the output of the final output and the intermediate outputs close to each other
        for i in range(1, len(inputs)):
            hard_loss += self.special_loss(inputs[i], target) * self.alpha
        hard_loss = hard_loss / ((len(inputs) - 1) * self.alpha + 1)
        if self.lamb > 0:
            for i in range(1, len(inputs)):
                soft_loss += SoftKL(inputs[i], finput, self.temperature)  # TODO: check the temperature
                soft_loss += SoftKL(finput, inputs[i],
                                              self.temperature)  # making the final output better # normalize the hard loss

            soft_loss = soft_loss / (2 * (len(inputs) - 1)+1e-8)  # normalize the soft loss
        return self.lamb * soft_loss + (1 - self.lamb) * hard_loss