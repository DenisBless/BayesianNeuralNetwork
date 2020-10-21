import numpy as np
import torch


class NegativeELBO(torch.nn.Module):
    """
    Returns value of the evidence lower bound (ELBO)
    the negative ELBO can be decomposed into the expected negative log likelihood ENLL and the KL-Divergence
    between the variational and the prior distribution
    """

    def __init__(self, net, loss=torch.nn.MSELoss()):
        super(NegativeELBO, self).__init__()
        self.loss = loss
        self.net = net

    def __call__(self, X, Y, beta):
        Y_pred = self.net(X)
        if np.isnan(Y_pred.detach().numpy()).any():
            self.net(torch.Tensor(1))
        assert not np.isnan(Y_pred.detach().numpy()).any()
        return self.calc(Y_pred, Y, beta)

    def get_kl_div(self):  #
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