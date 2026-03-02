"""
Download and extract UCSD from Kaggle.

This assumes:
1. All requirements are satisfied according to requirements.txt
2. Kaggle.json exists in ~/config/kaggle/kaggle.json
"""
import kagglehub
from absl import app


def main(argv):
    """Download data from Kaggle."""
    path = kagglehub.dataset_download("karthiknm1/ucsd-anomaly-detection-dataset")
    print(f"Downloaded and unzipped dataset to: {path}")

    path = kagglehub.dataset_download("hihnguynth/cuhk-avenue-dataset")
    print(f"Downloaded and unzipped dataset to: {path}")

    path = kagglehub.dataset_download("odins0n/ucf-crime-dataset")
    print(f"Downloaded and unzipped dataset to: {path}")


if __name__ == "__main__":
    app.run(main)
