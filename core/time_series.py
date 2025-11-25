"""
Processes a bike-sharing dataset and modifies its structure
along different dimensions for analysis.

Demonstrates several tensor manipulations using PyTorch, including
reshaping, transposing, and one-hot encoding. It loads a dataset, reformats the
data to group it in meaningful dimensional arrangements (e.g., daily patterns),
and extracts specific features for further processing.
"""
import argparse

import numpy as np
import torch
from absl import app

torch.set_printoptions(edgeitems=2, threshold=50, linewidth=75)


def main(argv):
    del argv
    # Get dataset
    bikes_numpy = np.loadtxt(
        "../testdata/hour-fixed.csv",
        dtype=np.float32,
        delimiter=",",
        skiprows=1,
        converters={1: lambda x: float(x[8:10])})  # converts date strings to numbers
    bikes = torch.from_numpy(bikes_numpy)
    print(bikes.shape)  # {17520, 17}

    # Convert N x L x C
    daily_bikes = bikes.view(-1, 24, bikes.shape[1])  # bikes.shape[1] = 17
    # shape: {730, 24, 17} stride: {488, 17, 1}
    # original: 17520 * 17
    # new:      N * 24 * 17
    # N * 24 * 17 = 17520 * 17
    # N = 730
    # Stride is 17 * 24 = 480
    # -1 => infer dimensions automatically
    # Essentially break all data by 24

    # Get from {730, 24, 17} to {730, 17, 24}
    daily_bikes = daily_bikes.transpose(1, 2)
    # {730, 17, 24}
    print(daily_bikes.shape)

    # See first day
    first_day = bikes[:24].long()  # {24, 17}
    weather_onehot = torch.zeros(first_day.shape[0], 4)  # {24, 4}
    first_day[:, 9]  # 9 is whether situation 1:clear, 2:mist, 3:light rain...
    weather_onehot.scatter_(
        dim=1,
        index=first_day[:, 9].unsqueeze(1).long() - 1,
        value=1.0)
    print(weather_onehot)

    # Concatenate weather and bike data
    # torch.cat allocates a new storage
    torch.cat((bikes[:24], weather_onehot), 1)[:1]


if __name__ == "__main__":
    app.run(main)
