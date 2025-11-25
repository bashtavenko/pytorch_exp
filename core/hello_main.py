# Smoke test to make sure that the local Pytorch works
import torch
from absl import app
from absl import logging


def main(argv):
    del argv  # Unused.

    logging.info("PyTorch version: %s", torch.__version__)
    a = torch.ones(3, 3)
    b = torch.ones(3, 3)
    logging.info("CPU result:\n%s", a + b)

    # GPU check (only if CUDA is available)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        a = a.to(device)
        b = b.to(device)
        logging.info("CUDA device: %s", torch.cuda.get_device_name(device))
        logging.info("GPU result:\n%s", a + b)
    else:
        logging.info("CUDA is not available; skipping GPU test.")


if __name__ == '__main__':
    app.run(main)
