"""Same `temp_by_hand.py` but with nn.Linear"""
from absl import app
import torch
import torch.nn as nn
import torch.optim as optim


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()


def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val,
                  t_c_train, t_c_val):
    for epoch in range(1, n_epochs + 1):
        t_p_train = model(t_u_train)
        loss_train = loss_fn(t_p_train, t_c_train)

        t_p_val = model(t_u_val)
        loss_val = loss_fn(t_p_val, t_c_val)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                  f" Validation loss {loss_val.item():.4f}")


def main(argv):
    del argv

    t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
    t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
    t_c = torch.tensor(t_c).unsqueeze(1)  # adds the extra dimension at axis 1 for batch
    t_u = torch.tensor(t_u).unsqueeze(1)  # because nn.Linear expects a 2D tensor

    n_samples = t_u.shape[0]
    n_val = int(0.2 * n_samples)

    shuffled_indices = torch.randperm(n_samples)

    train_indices = shuffled_indices[:-n_val]
    val_indices = shuffled_indices[-n_val:]

    train_indices, val_indices

    t_u_train = t_u[train_indices]
    t_c_train = t_c[train_indices]

    t_u_val = t_u[val_indices]
    t_c_val = t_c[val_indices]

    t_un_train = 0.1 * t_u_train
    t_un_val = 0.1 * t_u_val

    # Linear model
    linear_model = nn.Linear(1, 1)
    print(f'weight: {linear_model.weight}, bias: {linear_model.bias}')

    optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)

    training_loop(
        n_epochs=3000,
        optimizer=optimizer,
        model=linear_model,
        loss_fn=loss_fn,
        t_u_train=t_un_train,
        t_u_val=t_un_val,
        t_c_train=t_c_train,
        t_c_val=t_c_val)

    print(linear_model.weight)
    print(linear_model.bias)


if __name__ == '__main__':
    app.run(main)
