from pathlib import Path
from typing import Annotated

import polars as pl
from cyclopts import App, Parameter
from hafnia.dataset.benchmark.benchmark import metric_calculations, run_inference_on_dataset
from hafnia.dataset.dataset_names import SampleField, SplitName
from hafnia.dataset.hafnia_dataset import HafniaDataset, Optional
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger
from hafnia.utils import get_dataset_path_in_hafnia_cloud, is_hafnia_cloud_job

from trainer_object_detection import utils
from trainer_object_detection.wrapped_model import InferenceConfig, WrappedModel

app = App(name="benchmark", help="Benchmark")

CLASS_MAPPING_OPTIONS = [None, *utils.CLASS_MAPPINGS.keys()]

""" Benchmarking examples
# Example: Benchmark pretrained model (RFDETRNano) for a vehicle detection task.
# The tricky part for this benchmark is that RFDETRNano is pretrained on coco datasets lables while the
# dataset have different labels. To solve this we needs to remap both the dataset and model predictions to
# a common label space. In this example we remap both to a common "vehicle detection"
# label space, but other remapping strategies are also possible.
python scripts/benchmark.py --model-class-mapping COCO2OnlyVehicle


hafnia experiment create --recipe-id 8618234d-b4da-4aa9-bb3e-3be86bb50369 --trainer-path . --cmd "python scripts/benchmark.py --model-class-mapping COCO2OnlyVehicle"

"""


@app.default
def main(
    model_path: Annotated[
        str,
        Parameter(
            help=(
                "Path to the trained model archive (.zip). Note: this is ignored when a checkpoint is "
                "available (e.g. a checkpoint selected for the experiment on the Hafnia platform) - the "
                "checkpoint is benchmarked instead of this model."
            )
        ),
    ] = "./pretrained_models/RFDETRNano.zip",
    inference: Annotated[Optional[InferenceConfig], Parameter(help="Inference configuration for the model")] = None,
    model_class_mapping: Annotated[
        Optional[str],
        Parameter(
            help=(
                "Class mapping applied to the model predictions to remap them into a common label space "
                f"with the ground truth. Options: {CLASS_MAPPING_OPTIONS}"
            )
        ),
    ] = None,
    dataset_class_mapping: Annotated[
        Optional[str],
        Parameter(
            help=(
                "Class mapping applied to the dataset ground-truth labels to remap them into a common "
                f"label space with the predictions. Options: {CLASS_MAPPING_OPTIONS}"
            )
        ),
    ] = None,
    split_name: Annotated[str, Parameter(help="Dataset split to run on")] = SplitName.TEST,
    save_annotations: Annotated[
        bool,
        Parameter(
            help="Write the predictions (annotations only, no image data) to the experiment artifacts folder."
        ),
    ] = True,
    samples: Annotated[
        Optional[int],
        Parameter(help="Limit the number of samples to run on. Useful for faster testing."),
    ] = None,
    ensemble_extra: Annotated[
        Optional[list[str]],
        Parameter(
            help=(
                "Extra model archive (.zip) paths to ensemble with the primary model/checkpoint. "
                "Each runs inference and all are fused via WBF into ONE prediction file — a single "
                "inference pipeline (AI City Track 6 ensemble rule)."
            )
        ),
    ] = None,
    tta_hflip: Annotated[
        bool,
        Parameter(help="Add a horizontal-flip TTA view for every ensemble member (doubles inference; fused via WBF)."),
    ] = False,
    wbf_iou: Annotated[float, Parameter(help="WBF IoU threshold for clustering boxes across members/views.")] = 0.55,
    wbf_skip_box_thr: Annotated[float, Parameter(help="Drop input boxes below this confidence before WBF.")] = 0.0,
    wbf_conf_type: Annotated[
        str, Parameter(help="WBF confidence fusion: 'box_and_model_avg' | 'avg' | 'max'.")
    ] = "box_and_model_avg",
):
    """Run a model on a Hafnia dataset split and compute detection metrics when ground truth is available.

    Loads the dataset (the hidden dataset when running on the Hafnia platform, otherwise a public
    sample dataset), runs the model on the requested split, and - when the split has ground-truth
    annotations - computes detection metrics and logs them through ``HafniaLogger``. When the split
    has no ground truth (e.g. a held-out test set without labels) the metric step is skipped, so the
    same script can also be used as a pure inference pass.

    The ``model_class_mapping`` and ``dataset_class_mapping`` flags project predictions and/or
    ground truth into a common label space, which is needed when a pretrained model (e.g. trained on
    COCO) is benchmarked against a dataset with a different label space. When ``save_annotations`` is
    set (default), the dataset with predictions appended as a new prediction task on each sample is
    written - annotations only, no image data - to the experiment artifacts folder for downstream
    analysis or visualization.
    """
    inference = inference or InferenceConfig()
    # Fail fast (before loading the dataset / running hours of inference) on bad ensemble args.
    if wbf_conf_type not in ("box_and_model_avg", "avg", "max"):
        raise ValueError(f"wbf_conf_type must be 'box_and_model_avg' | 'avg' | 'max', got '{wbf_conf_type}'")
    for _extra in ensemble_extra or []:
        if not Path(_extra).exists():
            raise FileNotFoundError(f"ensemble_extra model archive not found: {_extra}")
    logger = HafniaLogger(project_name="Benchmarking RF-DETR")
    if is_hafnia_cloud_job():  # For hafnia cloud execution
        path_dataset = get_dataset_path_in_hafnia_cloud()  # The path to the full/hidden dataset is returned
        dataset = HafniaDataset.from_path(path_dataset)
    else:
        # The small/public sample dataset is returned by name
        dataset = HafniaDataset.from_name("midwest-vehicle-detection", version="1.0.0")

    # Prefer a user-selected checkpoint over the configured model when one is available.
    checkpoint_model_path = utils.get_checkpoint_if_available(logger)
    if checkpoint_model_path is not None:
        user_logger.info(f"Using checkpoint '{checkpoint_model_path.name}' instead of '{model_path}'")
        model_path = checkpoint_model_path.as_posix()

    model = WrappedModel.load_model(model_path, inference_config=inference)
    model.optimize_for_inference()

    dataset_split = dataset.create_split_dataset(split_name=split_name)
    dataset_task_info = dataset.info.get_task_by_primitive(model.task.primitive)

    configuration = {
        "model": model.__class__.__name__,
        "compile": inference.compile,
        "batch_size": inference.batch_size,
        "threshold": inference.threshold,
        "dataset": dataset.info.dataset_name,
        "dataset_version": dataset.info.version,
        "model_filename": Path(model_path).name,
        "num_samples": len(dataset_split),
        "class_mapping_model": model_class_mapping,
        "class_mapping_dataset": dataset_class_mapping,
        "split_name": split_name,
        "num_select": inference.num_select,
        "resolution": inference.resolution,
        "grayworld": inference.grayworld,
        "ensemble_extra": ensemble_extra or [],
        "tta_hflip": tta_hflip,
        "wbf": {"iou": wbf_iou, "skip_box_thr": wbf_skip_box_thr, "conf_type": wbf_conf_type}
        if (ensemble_extra or tta_hflip)
        else None,
    }
    logger.log_configuration(configuration)

    if samples is not None:
        dataset_split = dataset_split.select_samples(n_samples=samples, seed=42)

    # Remap ground-truth classes to a common label space before inference if requested
    if dataset_class_mapping is not None:
        dataset_split = dataset_split.class_mapper(
            class_mapping=utils.CLASS_MAPPINGS[dataset_class_mapping],
            method="remove_undefined",
            task_name=model.task.name,
        )

    # Inference. Predictions are appended as a new task on each sample.
    prediction_post_fix = "/predictions"
    drop_columns = [SampleField.FILE_PATH, SampleField.VIDEO_INFO, SampleField.CAMERA_INFO, SampleField.META]
    extra_models = ensemble_extra or []
    do_ensemble = bool(extra_models) or tta_hflip

    def _infer(mdl):
        dsp = run_inference_on_dataset(
            dataset=dataset_split, model=mdl, task_name_prediction_postfix=prediction_post_fix
        )
        # Remap model prediction classes into the dataset's label space if requested
        if model_class_mapping is not None:
            dsp = dsp.class_mapper(
                class_mapping=utils.CLASS_MAPPINGS[model_class_mapping],
                method="remove_undefined",
                task_name=f"{dataset_task_info.name}{prediction_post_fix}",
            )
        return dsp

    # Workaround (2026-06-27): the platform "Download Experiment Outputs" is non-functional
    # (artifact stays null, no download endpoint). The model output (/opt/ml/model) IS downloadable
    # via GET /experiments/{id}/model, so predictions are ALSO written there to retrieve them.
    if not do_ensemble:
        dataset_with_predictions = _infer(model)
        if save_annotations:
            dataset_with_predictions.samples = dataset_with_predictions.samples.drop(drop_columns, strict=False)
            dataset_with_predictions.write_annotations(logger.path_model())
            dataset_with_predictions.write_annotations(logger._path_artifacts())

        gt_column = dataset_task_info.primitive.column_name()
        no_gt_data = dataset_split.samples.select(pl.col(gt_column).list.len()).sum().item() == 0
        if no_gt_data:
            user_logger.warning("No ground-truth annotations found in the selected split. Skipping metric calculation.")
            return logger
        metrics = metric_calculations(
            prediction_dataset=dataset_with_predictions, prediction_task_name_postfix=prediction_post_fix
        )
        for metric_name, metric_value in metrics.items():
            utils.safe_log_metric(logger, metric_name, metric_value)
        return logger

    # ENSEMBLE / TTA — a SINGLE inference pipeline (Track 6 ensemble rule): infer each
    # (member, view), fuse via WBF into ONE prediction file. Metric calc is skipped here;
    # this path targets the no-GT benchmark/test split. Use single-model mode on GT splits.
    import shutil
    import tempfile

    from trainer_object_detection import wbf

    members = [(model_path, "primary")] + [(p, Path(p).stem) for p in extra_models]
    views = [False, True] if tta_hflip else [False]
    primary_class_names = [c.name for c in model.task.classes]
    tmp = Path(tempfile.mkdtemp(prefix="ensemble_"))
    try:
        member_jsonls = []
        for mi, (zip_path, label) in enumerate(members):
            for hflip in views:
                # Reuse the already-loaded primary only when its view truly matches
                # (original AND unflipped); otherwise build a fresh model for this view.
                if mi == 0 and hflip is False and not inference.hflip:
                    mdl = model
                else:
                    cfg = InferenceConfig(
                        compile=inference.compile,
                        batch_size=inference.batch_size,
                        threshold=inference.threshold,
                        num_select=inference.num_select,
                        resolution=inference.resolution,
                        grayworld=inference.grayworld,
                        hflip=hflip,
                    )
                    mdl = WrappedModel.load_model(zip_path, inference_config=cfg)
                    mdl.optimize_for_inference()
                # Guard: every member must share the primary's class space, else WBF would
                # fuse mismatched class_idx into a schema-valid but semantically wrong file.
                member_class_names = [c.name for c in mdl.task.classes]
                if member_class_names != primary_class_names:
                    raise ValueError(
                        f"ensemble member '{label}' class set differs from primary "
                        f"({member_class_names} != {primary_class_names}); aborting to avoid a wrong submission."
                    )
                dsp = _infer(mdl)
                mdir = tmp / f"m{mi}_{label}_h{int(hflip)}"
                mdir.mkdir(parents=True, exist_ok=True)
                dsp.samples = dsp.samples.drop(drop_columns, strict=False)
                dsp.write_annotations(mdir)
                member_jsonls.append(str(mdir / "annotations.jsonl"))
                user_logger.info(f"[ensemble] inferred member='{label}' hflip={hflip}")

        out_dir = Path(logger.path_model())
        out_dir.mkdir(parents=True, exist_ok=True)
        stats = wbf.fuse_annotation_files(
            member_jsonls,
            str(out_dir / "annotations.jsonl"),
            weights=None,
            iou_thr=wbf_iou,
            skip_box_thr=wbf_skip_box_thr,
            conf_type=wbf_conf_type,
        )
        if stats.get("missing_secondary_keys"):
            user_logger.warning(f"[ensemble] {stats['missing_secondary_keys']} primary images missing from a member set")
        src_di = Path(member_jsonls[0]).parent / "dataset_info.json"
        if src_di.exists():
            shutil.copyfile(src_di, out_dir / "dataset_info.json")
        art_dir = Path(logger._path_artifacts())
        art_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_dir / "annotations.jsonl", art_dir / "annotations.jsonl")
        if (out_dir / "dataset_info.json").exists():
            shutil.copyfile(out_dir / "dataset_info.json", art_dir / "dataset_info.json")
        user_logger.info(f"[ensemble] WBF fused {len(member_jsonls)} prediction sets: {stats}")
        return logger
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main'
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=utils.CLI_TOOL)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")

    app()
