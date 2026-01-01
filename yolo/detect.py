"""
YOLO11 object detection with pretrained weights.
"""

from __future__ import annotations

import os
from absl import app
from absl import flags
from absl import logging

from ultralytics import YOLO

FLAGS = flags.FLAGS

flags.DEFINE_string("weights", "yolo11n.pt", "YOLO11 weights to use (e.g., yolo11n.pt, yolo11s.pt).")
flags.DEFINE_string("source", "../testdata/coco/000000142472.jpg", "Image/video path, directory, glob, or URL.")
flags.DEFINE_integer("imgsz", 640, "Inference image size.")
flags.DEFINE_float("conf", 0.25, "Confidence threshold.")
flags.DEFINE_string("device", "", "Device to run on: '', 'cpu', '0', '0,1', etc.")
flags.DEFINE_string("project", "output", "Where to save runs (relative to the directory).")
flags.DEFINE_string("name", "yolo11_detect", "Run name (subdir under --project).")


def main(argv: list[str]) -> None:
    del argv  # Unused.
    logging.info("Loading YOLO model weights: %s", FLAGS.weights)
    model = YOLO(FLAGS.weights)

    # Keep outputs inside the repo (instead of default ./runs)
    project_dir = os.path.normpath(FLAGS.project)

    logging.info(
        "Predicting: source=%s imgsz=%d conf=%.2f device=%s project=%s name=%s",
        FLAGS.source, FLAGS.imgsz, FLAGS.conf, FLAGS.device, project_dir, FLAGS.name
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

    # Basic summary logging
    if results:
        r0 = results[0]
        n = 0 if r0.boxes is None else len(r0.boxes)
        logging.info("Detections in first item: %d", n)
        if getattr(r0, "save_dir", None):
            logging.info("Saved outputs to: %s", r0.save_dir)
        else:
            logging.info("Saved outputs under: %s/%s", project_dir, FLAGS.name)


if __name__ == "__main__":
    app.run(main)