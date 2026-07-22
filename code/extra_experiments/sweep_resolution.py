"""Inference-resolution sweep on the source-domain VAL split (has GT → mAP).

Runs ONE fixed 704-trained checkpoint at several square inference resolutions and
records in-domain (source-val) detection metrics at each. This is supplementary
in-domain evidence for the paper's inference-time resolution-scaling lever; it does
NOT produce cross-city / hidden-benchmark scores (that server is closed) and it
extracts only aggregate numbers + predicted annotations — no image data — so it is
compliant with the platform's no-image-extraction rule.

Usage (Hafnia cloud):
  python scripts/sweep_resolution.py --model-path ./pretrained_models/run1_best.zip \
      --resolutions "704,832,896,1024,1120,1280"
"""
import gc
import json
from pathlib import Path
from typing import Annotated, Optional

import polars as pl
import torch
from cyclopts import App, Parameter
from hafnia.dataset.benchmark.benchmark import metric_calculations, run_inference_on_dataset
from hafnia.dataset.dataset_names import SplitName
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.experiment import HafniaLogger
from hafnia.log import user_logger
from hafnia.utils import get_dataset_path_in_hafnia_cloud, is_hafnia_cloud_job

from trainer_object_detection import utils
from trainer_object_detection.wrapped_model import InferenceConfig, WrappedModel

app = App(name="sweep", help="Inference-resolution sweep on the VAL split")


@app.default
def main(
    model_path: Annotated[str, Parameter(help="Trained model archive (.zip)")] = "./pretrained_models/run1_best.zip",
    resolutions: Annotated[str, Parameter(help="Comma-separated square resolutions")] = "704,832,896,1024,1120,1280",
    split_name: Annotated[str, Parameter(help="Split with GT to evaluate on")] = SplitName.VAL,
    threshold: Annotated[float, Parameter(help="Score threshold (low → full PR curve for mAP)")] = 0.01,
    num_select: Annotated[int, Parameter(help="Top-k detections per image")] = 300,
    samples: Annotated[Optional[int], Parameter(help="Optional cap on #images (debug)")] = None,
):
    res_list = [int(r) for r in resolutions.split(",") if r.strip()]
    for r in res_list:
        if r % 32 != 0:
            raise ValueError(f"resolution {r} is not a multiple of 32 (RF-DETR requirement)")

    logger = HafniaLogger(project_name="Resolution sweep RF-DETR (source-val)")
    if is_hafnia_cloud_job():
        dataset = HafniaDataset.from_path(get_dataset_path_in_hafnia_cloud())
    else:
        dataset = HafniaDataset.from_name("midwest-vehicle-detection", version="1.0.0")

    # This sweep intentionally uses the packaged --model-path (the fixed 704-trained base),
    # not a platform-attached checkpoint, so all resolutions share ONE checkpoint.
    dataset_split = dataset.create_split_dataset(split_name=split_name)
    if samples is not None:
        dataset_split = dataset_split.select_samples(n_samples=samples, seed=42)

    # Load once up front to resolve the GT column and fail fast if the split has no labels.
    probe = WrappedModel.load_model(model_path, inference_config=InferenceConfig(compile=False))
    gt_column = dataset.info.get_task_by_primitive(probe.task.primitive).primitive.column_name()
    has_gt = dataset_split.samples.select(pl.col(gt_column).list.len()).sum().item() > 0
    if not has_gt:
        raise RuntimeError(f"split '{split_name}' has no ground truth; cannot compute mAP")
    del probe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {}
    for r in res_list:
        cfg = InferenceConfig(compile=False, batch_size=1, threshold=threshold, num_select=num_select, resolution=str(r))
        model = WrappedModel.load_model(model_path, inference_config=cfg)
        model.optimize_for_inference()
        dsp = run_inference_on_dataset(
            dataset=dataset_split, model=model, task_name_prediction_postfix="/predictions"
        )
        metrics = metric_calculations(prediction_dataset=dsp, prediction_task_name_postfix="/predictions")
        metrics = {k: (float(v) if isinstance(v, (int, float)) or hasattr(v, "__float__") else v)
                   for k, v in metrics.items()}
        results[str(r)] = metrics
        for mk, mv in metrics.items():
            if isinstance(mv, (int, float)):
                utils.safe_log_metric(logger, f"res{r}/{mk}", mv)
        user_logger.info(f"[sweep] res={r}: {json.dumps(metrics, default=str)[:400]}")
        del model, dsp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(logger.path_model())
    out.mkdir(parents=True, exist_ok=True)
    payload = {"model_path": Path(model_path).name, "split": split_name, "threshold": threshold,
               "num_select": num_select, "num_images": len(dataset_split), "results": results}
    (out / "resolution_sweep.json").write_text(json.dumps(payload, indent=2, default=str))
    user_logger.info(f"[sweep] wrote {out / 'resolution_sweep.json'}")
    return logger


if __name__ == "__main__":
    app()
