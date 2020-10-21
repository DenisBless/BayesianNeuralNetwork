import math
import torch
import torch.nn.functional as F


class BayesLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, prior_std_init=0.05,
                 std_init=0.05) -> None:
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std_init = prior_std_init
        self.std_init = std_init

        self.w_mean = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.w_std = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.w_mean, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.w_mean)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self.w_std.data.uniform_(-self.std_init, self.std_init)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        act_mean = F.linear(input, self.w_mean, self.bias)
        act_std = torch.sqrt(F.linear(input ** 2, self.w_std ** 2))
        # rsample() enables the reparameterization trick
        eps = torch.distributions.Normal(loc=torch.zeros_like(act_std),
                                         scale=torch.ones_like(act_std)).rsample()

        return act_mean + act_std * eps

    def kl_div(self):
        # D_KL(q||p) where p ~ N(0,1)
        a = torch.log(torch.ones_like(self.w_std) * self.prior_std_init ** 2) - torch.log(self.w_std ** 2)
        b = (self.w_std ** 2 + self.w_mean ** 2) / self.prior_std_init ** 2
        return torch.sum(0.5 * (a + b - 1))
