"""
Download a small COCO subset with cars using FiftyOne and export images into testdata/.
"""

from __future__ import annotations

import os
from absl import app
from absl import flags
from absl import logging

import fiftyone as fo
import fiftyone.zoo as foz

FLAGS = flags.FLAGS

flags.DEFINE_string("split", "validation", "COCO split: 'train' or 'validation'")
flags.DEFINE_integer("max_samples", 20, "How many samples to download")
flags.DEFINE_string("out_dir", "../testdata/coco/", "Where to export images")
flags.DEFINE_bool("also_include_truck_bus", True, "Include truck/bus along with car")


def _find_detections_field(dataset: fo.Dataset) -> str:
    """
    Returns the name of the field that stores `fo.Detections` labels.
    """
    schema = dataset.get_field_schema()
    for name, field in schema.items():
        doc_type = getattr(field, "document_type", None)
        if doc_type is not None and issubclass(doc_type, fo.Detections):
            return name

    # Common fallback
    if "ground_truth" in schema:
        return "ground_truth"

    raise ValueError(f"No Detections label field found. Available fields: {list(schema.keys())}")


def main(argv: list[str]) -> None:
    del argv

    classes = ["car"]
    if FLAGS.also_include_truck_bus:
        classes += ["truck", "bus"]

    out_dir = os.path.normpath(FLAGS.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    logging.info("Loading COCO subset: split=%s classes=%s max_samples=%d", FLAGS.split, classes, FLAGS.max_samples)

    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=FLAGS.split,
        label_types=["detections"],
        classes=classes,
        max_samples=FLAGS.max_samples,
        shuffle=True,
        dataset_name=f"coco_{FLAGS.split}_{'_'.join(classes)}_{FLAGS.max_samples}",
    )

    label_field = _find_detections_field(dataset)
    logging.info("Using label field: %s", label_field)

    logging.info("Exporting images to: %s", out_dir)
    dataset.export(
        export_dir=out_dir,
        dataset_type=fo.types.COCODetectionDataset,
        label_field=label_field,
        classes=classes,
    )

    logging.info("Done. Export dir contains images/ and labels.json (COCO format).")


if __name__ == "__main__":
    app.run(main)