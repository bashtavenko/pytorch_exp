"""
Example of tabular data
"""

import numpy as np
import torch

torch.set_printoptions(edgeitems=2, precision=2, linewidth=75)
from absl import app
import csv


def main(argv):
    del argv
    wine_path = 'testdata/winequality-white.csv'
    wine_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";",
                            skiprows=1)
    print(wine_numpy)  # data

    # Show column list and shapes
    col_list = next(csv.reader(open(wine_path), delimiter=';'))
    print(wine_numpy.shape, col_list)

    # Convert to tensor
    wine_t = torch.from_numpy(wine_numpy)
    print(wine_t)

    # Quality is 1 ... 9
    # All columns, last row
    # The first one is the slice
    data = wine_t[:, :-1]  # all rows, excluding the last column (quality)
    # Only the last column
    target = wine_t[:, -1]  # all rows, quality column
    print(data.shape, target.shape)

    # One-hot encoding
    # scatter_ want integer indices, not float
    target = target.long()
    print(target)
    target_onehot = torch.zeros(target.shape[0], 10)
    # Modifies tensor in place
    target_onehot.scatter_(1, target.unsqueeze(1), 1.0)  # Extra dummy dimension

    random_indices = torch.randint(0, len(target), (5,))
    print("5 random wine quality ratings (original):")
    print(target[random_indices])

    print("\nFirst 5 wine quality ratings (one-hot encoded):")
    print(target_onehot[random_indices])

    # Categorical data
    data_mean = torch.mean(data, dim=0)
    print(f'data_mean: {data_mean}')
    data_var = torch.var(data, dim=0)
    print(f'data_var: {data_var}')
    data_normalized = (data - data_mean) / torch.sqrt(data_var)
    print(f'data_normalized: {data_normalized}')

    # Threshold
    bad_indexes = target <= 3
    print(f'shape: {bad_indexes.shape}, type: {bad_indexes.dtype}, sum: {bad_indexes.sum()}')
    bad_data = data[bad_indexes]  # shape is {20, 11} 20 is the number of bad wines

    bad_data = data[target <= 3]
    mid_data = data[(target > 3) & (target < 7)]  # <1>
    good_data = data[target >= 7]

    bad_mean = torch.mean(bad_data, dim=0)
    mid_mean = torch.mean(mid_data, dim=0)
    good_mean = torch.mean(good_data, dim=0)
    print(f'bad_mean: {bad_mean}, mid_mean: {mid_mean}, good_mean: {good_mean}')
    for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
        print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))


if __name__ == '__main__':
    app.run(main)
