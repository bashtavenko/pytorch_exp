"""Example of training a linear regression model with Autograd. """
import torch

torch.set_printoptions(edgeitems=2, linewidth=75)
from absl import app


def model(t_u, w, b):
    return w * t_u + b


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()


def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        # Explicitly zero the gradients before running the backward pass.
        # Could be called at any point in the loop before the backward pass.
        if params.grad is not None:
            params.grad.zero_()

        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()

        with torch.no_grad():  # <2>
            params -= learning_rate * params.grad

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))

    return params


def main(argv):
    del argv
    t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0,
                        3.0, -4.0, 6.0, 13.0, 21.0])
    t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9,
                        33.9, 21.8, 48.4, 60.4, 68.4])
    t_un = 0.1 * t_u

    training_loop(
        n_epochs=5000,
        learning_rate=1e-2,
        params=torch.tensor([1.0, 0.0], requires_grad=True),
        t_u=t_un,  # <2>
        t_c=t_c)


if __name__ == '__main__':
    app.run(main)
