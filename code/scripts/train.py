from pathlib import Path
from typing import Annotated, Optional, Type

import polars as pl
import torch
from cyclopts import App, Parameter
from hafnia import utils as hafnia_utils
from hafnia.dataset.benchmark.benchmark import metric_calculations, run_inference_on_dataset
from hafnia.dataset.dataset_names import SampleField, SplitName
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Primitive
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger
from rfdetr import detr

import trainer_object_detection.wrapped_model
from trainer_object_detection import utils
from trainer_object_detection.aug_presets import AUG_PRESETS, resolve_aug_preset
from trainer_object_detection.wrapped_model import InferenceConfig, InitModelConfig, WrappedModel

detr = utils.patch_to_support_experiment_tracker_with_hafnia(detr)

app = App(name="train", help="PyTorch Training")

MODEL_NAME_OPTIONS = [f"pretrained_models/{d.name}.zip" for d in trainer_object_detection.wrapped_model.MODEL_OPTIONS]

DEFAULT_INFERENCE_MODEL = "checkpoint_best_ema"
INFERENCE_MODEL_OPTIONS = [DEFAULT_INFERENCE_MODEL, "checkpoint_best_regular", "checkpoint_best_total"]


@app.default
def main(
    project_name: Annotated[str, Parameter(help="Project name for the experiment")] = "Trainer RF-DETR",
    model_path: Annotated[
        str,
        Parameter(
            help=(
                "Path to a compressed (zip) pretrained model used as the training starting point. "
                f"Options: {MODEL_NAME_OPTIONS}. Note: this is ignored when a checkpoint is available "
                "(e.g. a checkpoint selected for the experiment on the Hafnia platform) - training "
                "resumes from the checkpoint instead (and '--pretrained' is forced to True)."
            )
        ),
    ] = "./pretrained_models/RFDETRNano.zip",
    pretrained: Annotated[bool, Parameter(help="Initialize the model from pretrained weights")] = True,
    epochs: Annotated[int, Parameter(help="Number of epochs to train")] = 10,
    batch_size: Annotated[int, Parameter(help="Batch size for training")] = 8,
    grad_accumulation_steps: Annotated[
        int,
        Parameter(
            help="Number of gradient accumulation steps (effective batch size = batch_size * grad_accumulation_steps)"
        ),
    ] = 1,
    learning_rate: Annotated[float, Parameter(help="Learning rate for the optimizer (decoder LR)")] = 0.001,
    warmup_epochs: Annotated[float, Parameter(help="Cosine-schedule warmup length in epochs")] = 1.0,
    lr_encoder: Annotated[float, Parameter(help="Encoder (backbone) learning rate")] = 1.0e-4,
    lr_min_factor: Annotated[float, Parameter(help="Cosine minimum-LR factor")] = 0.05,
    resolution: Annotated[
        Optional[int],
        Parameter(help="Input resolution (square side in pixels). Defaults to each model's built-in value."),
    ] = None,
    task_name: Annotated[
        Optional[str],
        Parameter(
            help=(
                "Dataset task name used for training. Only required when the dataset has multiple tasks "
                "matching the model primitive."
            )
        ),
    ] = None,
    samples: Annotated[
        Optional[int],
        Parameter(help="Number of samples to use for training (omit to use all samples). Use for testing purposes."),
    ] = None,
    stop_early: Annotated[
        bool,
        Parameter(
            help=(
                "Exit before training starts. Can be used to avoid long training times when smoke-testing the pipeline."
            )
        ),
    ] = False,
    inference_model_name: Annotated[
        str,
        Parameter(
            help=(
                f"Checkpoint used for the post-training benchmark on the test split. Options: {INFERENCE_MODEL_OPTIONS}"
            )
        ),
    ] = DEFAULT_INFERENCE_MODEL,
    inference_config: Annotated[
        Optional[InferenceConfig], Parameter(help="Inference configuration used for the post-training benchmark")
    ] = None,
    aug_preset: Annotated[
        str,
        Parameter(help=f"Cross-city DG augmentation preset. Options: {list(AUG_PRESETS)}"),
    ] = "dg_crosscity",
    augmentation_backend: Annotated[
        str,
        Parameter(help="Augmentation backend: 'cpu' (Albumentations, always available) or 'gpu'/'auto' (needs Kornia, NOT bundled). Default 'cpu'."),
    ] = "cpu",
    num_workers: Annotated[
        int,
        Parameter(help="Dataloader workers; raise to relieve the data-loading bottleneck on multi-vCPU nodes"),
    ] = 4,
    pseudo_label: Annotated[
        bool,
        Parameter(
            help=(
                "Self-training: before training, run the model from --model-path as a teacher over the "
                "unlabeled TEST split, keep confident predictions as pseudo ground truth, and add those "
                "images to the training set. Fully automatic (no manual labels). "
                "The teacher inference uses --inference-config (resolution/grayworld/num_select apply). "
                "NOT used for any reported L0-L4 result; kept as an exploratory research path."
            )
        ),
    ] = False,
    pseudo_thresholds: Annotated[
        str,
        Parameter(
            help=(
                "Per-class confidence thresholds for pseudo labels as 'default=0.5,Person=0.65,...'. "
                "Class names must match the dataset task classes; 'default' applies to unlisted classes."
            )
        ),
    ] = "default=0.50,Person=0.65",
    pseudo_min_boxes: Annotated[
        int,
        Parameter(
            help=(
                "Skip pseudo images with fewer confident boxes than this. Sparse pseudo images are mostly "
                "teacher misses turned into false background supervision (recall poisoning)."
            )
        ),
    ] = 2,
    pseudo_max_images: Annotated[
        Optional[int],
        Parameter(help="Optional cap on pseudo-labeled images added to training (keeps epoch time bounded)."),
    ] = None,
):
    """Train an RF-DETR object detection model on a Hafnia dataset.

    Loads the selected managed Hafnia training dataset when running on the platform; when
    executed locally, the small public sample dataset is used. (The evaluation benchmark is a
    separate held-out dataset and is never loaded here.) Initializes an RF-DETR model from
    the compressed model archive pointed to by ``model_path`` (optionally with pretrained weights), converts the
    train/val splits to COCO format and runs RF-DETR training.

    After training, every ``checkpoint_*.pth`` produced by RF-DETR is repackaged as a
    standalone compressed Hafnia model archive (weights + serialized model config bundled into a
    single ``.zip``) under the experiment model and checkpoints folders. The checkpoint selected
    by ``inference_model_name`` is then
    loaded as a ``WrappedModel``, optimized for inference (e.g. ``torch.compile`` when enabled
    via ``inference_config``) and run on the held-out test split. Predictions are written to
    the experiment artifacts folder. When the test split has ground-truth annotations, detection
    metrics are computed via ``metric_calculations`` and logged through ``HafniaLogger``; if no
    ground truth is present the metric step is skipped with a warning.
    """
    inference_config = inference_config or InferenceConfig()
    # Check cuda availability
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")

    logger = HafniaLogger(project_name=project_name)

    if hafnia_utils.is_hafnia_cloud_job():  # For hafnia cloud execution
        path_dataset = hafnia_utils.get_dataset_path_in_hafnia_cloud()  # managed training dataset
        dataset = HafniaDataset.from_path(path_dataset)
    else:
        dataset = HafniaDataset.from_name("eccv-cross-city", version="1.0.0")

    if samples is not None:
        dataset = dataset.select_samples(n_samples=samples)

    checkpoint_model_path = utils.get_checkpoint_if_available(logger)
    if checkpoint_model_path is not None:
        user_logger.info(f"Using checkpoint '{checkpoint_model_path.name}' as pretrained model")
        model_path = checkpoint_model_path.as_posix()
        # Resuming from a checkpoint always uses its weights, regardless of the '--pretrained' flag.
        pretrained = True

    model_config = InitModelConfig.load_model(model_path, use_weights=pretrained)
    model_primitive = model_config.task.primitive

    model_trainer = model_config.get_trainer()
    configuration = {
        "model": model_path,
        "pretrained": pretrained,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accumulation_steps": grad_accumulation_steps,
        "learning_rate": learning_rate,
        "resolution": resolution,
        "dataset": dataset.info.dataset_name,
        "has_cuda": has_cuda,
        "aug_preset": aug_preset,
        "augmentation_backend": augmentation_backend,
        "num_workers": num_workers,
    }

    if has_cuda:
        configuration["device"] = "cuda"
        configuration["num_gpus"] = torch.cuda.device_count()
        configuration["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    logger.log_configuration(configuration)

    task_info = get_dataset_task_from_model_primitive(dataset, model_primitive, task_name)

    dataset_test = dataset.create_split_dataset(split_name=SplitName.TEST)
    dataset_train_val = dataset.create_split_dataset(split_name=[SplitName.TRAIN, SplitName.VAL])
    dataset_train_val = remove_images_with_no_bboxes(dataset_train_val, model_primitive=model_primitive)

    if pseudo_label:
        dataset_train_val = add_pseudo_labeled_test_images(
            dataset_train_val=dataset_train_val,
            dataset_test=dataset_test,
            teacher_model_path=model_path,
            task_info=task_info,
            inference_config=inference_config,
            pseudo_thresholds=pseudo_thresholds,
            pseudo_min_boxes=pseudo_min_boxes,
            pseudo_max_images=pseudo_max_images,
        )

    # Convert dataset to COCO format for training
    dataset_name = dataset_train_val.info.dataset_name
    dataset_path = Path(".data") / f"format_coco_roboflow_{dataset_name}"
    dataset_train_val.to_coco_format(dataset_path, task_name=task_info.name)
    path_experiment = logger._local_experiment_path
    path_experiment.mkdir(parents=True, exist_ok=True)

    if stop_early:
        user_logger.info("Early stopping before training was activated with '--stop_early' flag.")
        return None

    # --- run2 res768 fix: enable backbone gradient checkpointing to fit 768px on a
    # 16GB T4. `gradient_checkpointing` is a ModelConfig field (NOT a TrainConfig
    # field), and TrainConfig has extra="ignore", so passing it to .train(**kwargs)
    # would be SILENTLY DROPPED. The training nn.Module is (re)built from this
    # model_config inside RFDETR.train() -> RFDETRModelModule -> build_model_from_config
    # -> _namespace_from_configs (forwards model_config.gradient_checkpointing) ->
    # build_backbone(gradient_checkpointing=...). The Large encoder is
    # dinov2_windowed_small, whose WindowedDinov2WithRegistersEncoder supports
    # checkpointing (the `assert not gradient_checkpointing` only guards the
    # NON-windowed Dinov2 path, which Large does not use). So mutating it here, before
    # .train(), is the correct and only injection point.
    model_trainer.model_config.gradient_checkpointing = True
    user_logger.info(
        f"gradient_checkpointing set to {model_trainer.model_config.gradient_checkpointing} "
        "on model_config (memory fit for high-resolution training)."
    )

    model_trainer.train(
        dataset_dir=dataset_path.as_posix(),
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        grad_accum_steps=grad_accumulation_steps,
        output_dir=path_experiment.as_posix(),
        resolution=resolution,
        aug_config=resolve_aug_preset(aug_preset),
        augmentation_backend=augmentation_backend,
        num_workers=num_workers,
        # --- LR-schedule fix (verified defect: defaults = step + lr_drop=100 +
        # warmup_epochs=0 => constant LR, no warmup, no decay at <=100 epochs).
        # Valid rfdetr TrainConfig fields forwarded via RFDETR.train(**kwargs);
        # confirm they took effect in the logged training config. ---
        lr_scheduler="cosine",
        # Warmup / encoder-LR / min-LR-factor are CLI arguments so the base run and
        # the warm-start fine-tune are reproducible from the command line. The base
        # 80-epoch run used a 1.0-epoch warmup (the default here); the later short
        # self-training run used 0.25.
        warmup_epochs=warmup_epochs,
        lr_min_factor=lr_min_factor,
        lr_encoder=lr_encoder,
    )

    model_folder_path = logger.path_model()
    # Repackage each final checkpoint as a single compressed model archive in the model folder
    # (e.g. "checkpoint_best_regular.zip" and "checkpoint_best_total.zip").
    final_models = list(path_experiment.glob("checkpoint_*.pth"))
    model_path = {}
    for checkpoint_path in final_models:
        model_name = checkpoint_path.stem  # e.g. "checkpoint_best_regular"
        model_checkpoint_path = model_folder_path / f"{model_name}.zip"
        model_config = InitModelConfig(name=model_config.name, task=task_info, model_weight_path=str(checkpoint_path))
        model_config.save_model(model_checkpoint_path)
        model_path[model_name] = model_checkpoint_path

    checkpoints_folder_path = logger.path_model_checkpoints()
    checkpoint_model_paths = final_models  # For now we simply add final models as checkpoints
    for ckpt_path in checkpoint_model_paths:
        model_config = InitModelConfig(name=model_config.name, task=task_info, model_weight_path=str(ckpt_path))
        model_config.save_model(checkpoints_folder_path / f"{ckpt_path.stem}.zip")

    #### 'TEST' split inference/benchmarking ####
    # Guarded: checkpoints are already saved above, so a failure in the post-training
    # inference/write (e.g. RAM exhaustion on Lite at high resolution — this exact stage
    # killed a finished 44h run) must not fail the experiment. On failure the checkpoints
    # remain retrievable and predictions can be regenerated with a separate benchmark run.
    try:
        import gc

        inference_model = WrappedModel.load_model(model_path[inference_model_name], inference_config=inference_config)
        inference_model.optimize_for_inference()

        dataset_with_predictions = run_inference_on_dataset(dataset=dataset_test, model=inference_model)

        # Free the model (host + CUDA memory) BEFORE the write: annotations serialization is
        # the peak-RAM moment on a 16GB instance and does not need the model anymore.
        del inference_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Experiment output folder
        path_experiment_output_folder = logger._path_artifacts()
        # Save predictions to experiment output folder (drops unneeded columns)
        drop_columns = [SampleField.FILE_PATH, SampleField.VIDEO_INFO, SampleField.CAMERA_INFO, SampleField.META]
        dataset_with_predictions.samples = dataset_with_predictions.samples.drop(drop_columns, strict=False)
        dataset_with_predictions.write_annotations(path_experiment_output_folder)
        # Also write to the downloadable model folder (artifacts folder has no download endpoint).
        dataset_with_predictions.write_annotations(logger.path_model())

        no_gt_data = dataset_test.samples.select(pl.col(task_info.primitive.column_name()).list.len()).sum().item() == 0
        if no_gt_data:  # Skip metric calculation for test sets without ground-truth annotations
            user_logger.warning("No ground-truth annotations found in the test set. Skipping metric calculation.")
            return logger

        metrics = metric_calculations(prediction_dataset=dataset_with_predictions)
        for metric_name, metric_value in metrics.items():
            utils.safe_log_metric(logger, metric_name, metric_value)
    except Exception as exc:  # noqa: BLE001 - training result must survive a failed benchmark step
        user_logger.error(f"Post-training inference/benchmark failed (checkpoints are saved): {exc}")

    return logger


def parse_pseudo_thresholds(spec: str, class_names: list) -> dict:
    """Parse 'default=0.5,Person=0.65' into {class_name: thr} with a 'default' fallback key.

    Raises on unknown class names so a typo fails at job start, not after hours of inference.
    """
    thr_map = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"pseudo-thresholds entry {token!r} must be 'Name=value'")
        name, value = token.rsplit("=", 1)
        name = name.strip()
        if name != "default" and name not in class_names:
            raise ValueError(f"pseudo-thresholds class {name!r} not in dataset classes {class_names}")
        thr_map[name] = float(value)
    if "default" not in thr_map:
        raise ValueError("pseudo-thresholds must include a 'default=<value>' entry")
    return thr_map


def add_pseudo_labeled_test_images(
    dataset_train_val: HafniaDataset,
    dataset_test: HafniaDataset,
    teacher_model_path: str,
    task_info: TaskInfo,
    inference_config: InferenceConfig,
    pseudo_thresholds: str,
    pseudo_min_boxes: int,
    pseudo_max_images: Optional[int],
) -> HafniaDataset:
    """Self-training stage: pseudo-label the unlabeled TEST split with the teacher and merge into training.

    Fully automatic (no manual labels). Confident teacher predictions are rewritten as ground-truth
    boxes on the dataset's task so `to_coco_format` picks them up like source GT. Images with fewer
    than `pseudo_min_boxes` confident boxes are dropped: a weak teacher's misses would otherwise
    become explicit background supervision and train the student NOT to detect (recall poisoning).
    """
    import gc
    import zipfile

    if not zipfile.is_zipfile(teacher_model_path):
        raise ValueError(
            f"[self-training] teacher '{teacher_model_path}' is not a valid zip archive "
            "(LFS pointer or missing file?) - aborting before any inference work."
        )

    class_names = [c.name for c in task_info.classes]
    thr_map = parse_pseudo_thresholds(pseudo_thresholds, class_names)
    default_thr = thr_map["default"]

    user_logger.info(
        f"[self-training] teacher={teacher_model_path} thresholds={thr_map} "
        f"min_boxes={pseudo_min_boxes} max_images={pseudo_max_images}"
    )
    teacher = WrappedModel.load_model(teacher_model_path, inference_config=inference_config)
    # Predictions are appended under the TEACHER's task name (+"/predictions"), which can
    # differ from the dataset task name; derive it from the model, not the dataset.
    teacher_task = teacher.get_model_info().tasks[0]
    teacher_class_names = [c.name for c in teacher_task.classes]
    if teacher_class_names != class_names:
        raise ValueError(
            f"[self-training] teacher classes {teacher_class_names} != dataset classes {class_names}"
        )
    prediction_task_name = f"{teacher_task.name}/predictions"
    try:
        teacher.optimize_for_inference()
        dataset_test_pred = run_inference_on_dataset(dataset=dataset_test, model=teacher)
    finally:
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        try:
            torch._dynamo.reset()
        except Exception:  # noqa: BLE001 - best-effort cache release
            pass

    # COCO export copies images by BASENAME and silently skips duplicates, so a pseudo image
    # sharing a basename with a source image would silently point at the wrong pixels.
    source_basenames = {
        Path(p).name for p in dataset_train_val.samples.get_column(SampleField.FILE_PATH).to_list() if p
    }

    schema = dataset_train_val.samples.schema
    pseudo_rows = []
    n_boxes_total = 0
    n_basename_collisions = 0
    for row in dataset_test_pred.samples.iter_rows(named=True):
        kept = []
        for box in row.get("bboxes") or []:
            if box.get("task_name") != prediction_task_name:
                continue
            confidence = box.get("confidence") or 0.0
            if confidence < thr_map.get(box.get("class_name"), default_thr):
                continue
            # Rewrite as ground truth on the dataset task so to_coco_format exports it.
            box["ground_truth"] = True
            box["task_name"] = task_info.name
            box["confidence"] = None
            kept.append(box)
        if len(kept) < pseudo_min_boxes:
            continue
        file_path = row.get("file_path")
        if file_path and Path(file_path).name in source_basenames:
            n_basename_collisions += 1
            continue
        row["bboxes"] = kept
        row["split"] = SplitName.TRAIN
        n_boxes_total += len(kept)
        pseudo_rows.append(row)
        if pseudo_max_images is not None and len(pseudo_rows) >= pseudo_max_images:
            break

    if n_basename_collisions:
        user_logger.warning(
            f"[self-training] dropped {n_basename_collisions} pseudo images whose basenames collide "
            "with source images (COCO export dedups by basename)."
        )

    if not pseudo_rows:
        raise ValueError("[self-training] produced 0 pseudo-labeled images - thresholds too strict?")

    user_logger.info(
        f"[self-training] kept {len(pseudo_rows)}/{len(dataset_test_pred.samples)} test images "
        f"({n_boxes_total} pseudo boxes, {n_boxes_total / len(pseudo_rows):.1f}/img)"
    )
    pseudo_df = pl.DataFrame(pseudo_rows, schema=schema).select(dataset_train_val.samples.columns)
    merged = pl.concat([dataset_train_val.samples, pseudo_df], how="vertical_relaxed")
    return dataset_train_val.update_samples(merged)


def remove_images_with_no_bboxes(dataset: HafniaDataset, model_primitive: Type[Primitive]) -> HafniaDataset:
    if not dataset.has_primitive(model_primitive):
        raise ValueError("Dataset does not contain bounding box information.")

    filter_column_name = model_primitive.column_name()
    samples_with_bboxes = dataset.samples.filter(pl.col(filter_column_name).list.len() > 0)
    dataset = dataset.update_samples(samples_with_bboxes)
    return dataset


def get_dataset_task_from_model_primitive(
    dataset: HafniaDataset,
    model_primitive: Type[Primitive],
    task_name: Optional[str] = None,
) -> TaskInfo:
    """Select the dataset task that matches the model primitive type."""

    # Get dataset tasks matching the model primitive
    matching_tasks = dataset.info.get_tasks_by_primitive(model_primitive)
    if len(matching_tasks) == 1:
        matching_task = matching_tasks[0]
        return matching_task

    if len(matching_tasks) == 0:
        available_primitives = [str(t.primitive.__name__) for t in dataset.info.tasks]
        raise ValueError(
            f"The selected model requires the dataset to have '{model_primitive}' annotations. "
            f"However, the dataset only contains the following primitives: {available_primitives}"
        )

    if task_name is None:
        matching_task_names = [t.name for t in matching_tasks]
        raise ValueError(
            f"The dataset contains multiple tasks with the required primitive '{model_primitive}'. "
            f"Please specify which task to use with the '--task_name' flag. "
            f"Matching tasks: {matching_task_names}"
        )

    model_task_info = dataset.info.get_task_by_name(task_name)

    if model_task_info.primitive != model_primitive:
        raise ValueError(f"The specified task '{task_name}' does not have the required primitive '{model_primitive}'.")

    return model_task_info


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main'
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=utils.CLI_TOOL, order=0)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")

    app()
