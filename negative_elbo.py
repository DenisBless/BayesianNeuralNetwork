import numpy as np
import torch


class NegativeELBO(torch.nn.Module):
    """
    Returns value of the negative evidence lower bound (ELBO).
    The negative ELBO can be decomposed into the expected negative log likelihood ENLL and the KL-Divergence
    between the variational and the prior distribution
    """
    def __init__(self, net, loss=torch.nn.MSELoss()):
        super(NegativeELBO, self).__init__()
        self.loss = loss
        self.net = net

    def __call__(self, X, Y, beta):
        Y_pred = self.net(X).reshape(-1,)
        assert Y_pred.shape == Y.shape, print(Y_pred.shape, print(Y.shape))
        return self.calc(Y_pred, Y, beta)

    def get_kl_div(self):
        kl = 0.0
        for module in self.net.modules():
            if hasattr(module, 'kl_div'):
                kl += module.kl_div()
        return kl

    def calc(self, Y_pred, Y, beta):
        ENLL = self.loss(Y_pred, Y)
        kl_div = self.get_kl_div()
        loss = ENLL + beta * kl_div
        return loss