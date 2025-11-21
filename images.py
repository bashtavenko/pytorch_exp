import torch
from absl import app
import imageio.v2 as imageio
import os


def main(argv):
    del argv
    img_arr = imageio.imread('testdata/silva.jpg')
    print(img_arr.shape)  # Gets (1406, 1599, 3) H x W x C

    # Convert to (C, H, W)
    img = torch.from_numpy(img_arr)
    img_t = img.permute(2, 0, 1)

    # Keep the first 3 channels
    # In case of RGB it is redundant, but if there were 4 it would keep the first 3
    img_t = img_t[:3]

    # Cat examples
    data_dir = 'testdata/image-cats'
    png_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
    batch = torch.zeros(len(png_files), 3, 256, 256, dtype=torch.uint8)

    for i, filename in enumerate(png_files):
        img_arr = imageio.imread(os.path.join(data_dir, filename))
        print(img_arr.shape)
        img_t = torch.from_numpy(img_arr)
        img_t = img_t.permute(2, 0, 1)
        img_t = img_t[:3]  # <1>
        batch[i] = img_t

    # Normalizing the image - simply dividing by 255.0
    batch = batch.float()
    batch /= 255.0

    # Normalizing image -- mean / std
    n_channels = batch.shape[1]
    for c in range(n_channels):
        # Get all rows with a channel c ...
        mean = torch.mean(batch[:, c])
        std = torch.std(batch[:, c])
        print(f"Image mean: {mean:.4f}, std: {std:.4f}")
        batch[:, c] = (batch[:, c] - mean) / std


if __name__ == '__main__':
    app.run(main)
