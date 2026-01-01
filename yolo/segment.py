"""
YOLO11 instance segmentation demo .
"""

from __future__ import annotations

import os
from absl import app
from absl import flags
from absl import logging

from ultralytics import YOLO

FLAGS = flags.FLAGS

flags.DEFINE_string("weights", "yolo11n-seg.pt", "YOLO11-seg weights (e.g., yolo11n-seg.pt).")
flags.DEFINE_string("source", "../testdata/coco/000000142585.jpg", "Image/video path, directory, glob, or URL.")
flags.DEFINE_integer("imgsz", 640, "Inference image size.")
flags.DEFINE_float("conf", 0.25, "Confidence threshold.")
flags.DEFINE_string("device", "", "Device to run on: '', 'cpu', '0', '0,1', etc.")
flags.DEFINE_string("project", "output", "Where to save runs (relative to the dir).")
flags.DEFINE_string("name", "yolo11_segment", "Run name (subdir under --project).")


def main(argv: list[str]) -> None:
    del argv  # Unused.
    logging.info("Loading YOLO segmentation model weights: %s", FLAGS.weights)
    model = YOLO(FLAGS.weights)

    project_dir = os.path.normpath(FLAGS.project)

    logging.info(
        "Segmenting: source=%s imgsz=%d conf=%.2f device=%s project=%s name=%s",
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

    if results:
        r0 = results[0]
        n_boxes = 0 if r0.boxes is None else len(r0.boxes)
        has_masks = (getattr(r0, "masks", None) is not None) and (getattr(r0.masks, "data", None) is not None)
        logging.info("Instances in first item: %d", n_boxes)
        logging.info("Masks available: %s", "yes" if has_masks else "no")
        if getattr(r0, "save_dir", None):
            logging.info("Saved outputs to: %s", r0.save_dir)
        else:
            logging.info("Saved outputs under: %s/%s", project_dir, FLAGS.name)


if __name__ == "__main__":
    app.run(main)