"""
License plate detection using a YOLOv11 model hosted on Hugging Face.

Model repo:
  https://huggingface.co/morsetechlab/yolov11-license-plate-detection

This script downloads the .pt weights from Hugging Face (into the HF cache)
and then loads them with Ultralytics YOLO.

In a real scenario it requires cropping the car detection box to this input.
"""

from __future__ import annotations

import os
from absl import app
from absl import flags
from absl import logging

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "repo_id",
    "morsetechlab/yolov11-license-plate-detection",
    "Hugging Face repo id containing the weights.",
)
flags.DEFINE_string(
    "filename",
    "license-plate-finetune-v1l.pt",
    "Weights filename inside the HF repo (e.g. license-plate-finetune-v1n.pt/v1s.pt/v1m.pt/v1l.pt/v1x.pt).",
)
flags.DEFINE_string("source", "../testdata/coco/000000142585.jpg", "Image/video path, directory, glob, or URL.")
flags.DEFINE_integer("imgsz", 640, "Inference image size.")
flags.DEFINE_float("conf", 0.25, "Confidence threshold.")
flags.DEFINE_string("device", "", "Device to run on: '', 'cpu', '0', '0,1', etc.")
flags.DEFINE_string("project", "output", "Where to save runs (relative to the directory).")
flags.DEFINE_string("name", "license_plate_detect", "Run name (subdir under --project).")


def main(argv: list[str]) -> None:
    del argv  # Unused.

    logging.info("Downloading weights from Hugging Face: %s/%s", FLAGS.repo_id, FLAGS.filename)
    weights_path = hf_hub_download(
        repo_id=FLAGS.repo_id,
        filename=FLAGS.filename,
    )
    logging.info("Weights cached at: %s", weights_path)

    logging.info("Loading YOLO model weights: %s", weights_path)
    model = YOLO(weights_path)

    project_dir = os.path.normpath(FLAGS.project)

    logging.info(
        "Predicting: source=%s imgsz=%d conf=%.2f device=%s project=%s name=%s",
        FLAGS.source,
        FLAGS.imgsz,
        FLAGS.conf,
        FLAGS.device,
        project_dir,
        FLAGS.name,
    )

    results = model.predict(
        source=FLAGS.source,
        imgsz=FLAGS.imgsz,
        conf=FLAGS.conf,
        device=FLAGS.device,
        project=project_dir,
        name=FLAGS.name,
        save=True,
        verbose=False,
    )

    if results:
        r0 = results[0]
        n = 0 if getattr(r0, "boxes", None) is None else len(r0.boxes)
        logging.info("Detections in first item: %d", n)
        if getattr(r0, "save_dir", None):
            logging.info("Saved outputs to: %s", r0.save_dir)
        else:
            logging.info("Saved outputs under: %s/%s", project_dir, FLAGS.name)


if __name__ == "__main__":
    app.run(main)