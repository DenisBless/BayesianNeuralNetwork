import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from bayesian_linear import BayesLinear
from negative_elbo import NegativeELBO
import torch.utils.data as Data


# Build the model
class BayesNN(torch.nn.Module):
    def __init__(self):
        super(BayesNN, self).__init__()
        self.bl1 = BayesLinear(1, 200)
        self.bl2 = BayesLinear(200, 100)
        self.bl3 = BayesLinear(100, 1)

    def forward(self, x):
        x = F.relu(self.bl1(x))
        x = F.relu(self.bl2(x))
        x = self.bl3(x)
        return x


# Build the model
class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.l1 = torch.nn.Linear(1, 100)
        self.l2 = torch.nn.Linear(100, 50)
        self.l3 = torch.nn.Linear(50, 20)
        self.l4 = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x


if __name__ == '__main__':
    # %%
    # Create the dataset
    x1 = np.linspace(-3, -1, 10)
    y1 = np.ones_like(x1)
    x2 = np.linspace(-1, 1, 20)
    y2 = np.zeros_like(x2)
    x3 = np.linspace(1, 3, 10)
    y3 = np.ones_like(x3)

    X = torch.Tensor(np.concatenate([x1, x2, x3])).reshape(-1, 1)
    Y = torch.Tensor(np.concatenate([y1, y2, y3]))

    plt.scatter(X, Y)

    torch_dataset = Data.TensorDataset(X, Y)
    BATCH_SIZE = 10
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2)

    # Train the model
    net = BayesNN()
    # net = torch.nn.Sequential(
    #     torch.nn.Linear(1, 200),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(200, 100),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(100, 1),
    # )
    neg_elbo = NegativeELBO(net)

    num_epochs = 1000

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(params=net.parameters(), lr=2e-4)

    for epoch in range(num_epochs):
        for i, (batch_x, batch_y) in enumerate(loader):  # for each training step
            # loss = neg_elbo(X, Y, 0)
            Y_pred = net(batch_x)
            loss = loss_fn(batch_y, Y_pred)
            optim.zero_grad()
            loss.backward()
            print(loss)
            optim.step()

        if epoch % 10 == 0:
            # Evaluate the model
            X_eval = torch.Tensor(np.linspace(-5, 5, 60)).reshape(-1, 1)
            Y_pred = net(X_eval)
            plt.scatter(X, Y)
            plt.plot(X_eval, Y_pred.detach().numpy())
            plt.show()
