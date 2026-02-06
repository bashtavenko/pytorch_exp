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

    # GPU or MPS check
    if torch.cuda.is_available():
        device = torch.device("cuda")
        a = a.to(device)
        b = b.to(device)
        logging.info("CUDA device")
        logging.info("GPU result:\n%s", a + b)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        a = a.to(device)
        b = b.to(device)
        logging.info("MPS device")
        logging.info("GPU result:\n%s", a + b)
    else:
        logging.info("CUDA or MPS are not available.")

if __name__ == '__main__':
    app.run(main)
