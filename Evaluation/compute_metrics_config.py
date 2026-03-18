"""Central registry of models/runs to include in evaluation scripts.

Both compute_segmentation_metrics.py and compute_junction_detection_metrics.py
import from here so that adding or removing a model only requires editing one
place.
"""

# WandB SAM run names and suffix for artifact to evaluate for segmentation and junction detection
SAM_MODELS_RUNS: list[str] = [
    # "SAM_LoRA_Finetuning_20260219_150640",
    "SAM_LoRA_Finetuning_20260224_154858",
]
SAM_PARAMS_ARTIFACT_SUFFIX = "_params_minloss:v0"

# nnUNet evaluations: list of (dataset_name, trainer_class) tuples.
# pre-computed predictions in NNUNET_RESULTS_DIR/<dataset_name>/<trainer_class>/NNUNET_PRED_DIR are evaluated
NNUNET_EVALUATIONS: list[tuple[str, str]] = [
    ("Dataset001_Segmentation_v1", "nnUNetTrainerWandb__nnUNetPlans__2d"),
]

# nnUNet models to evaluate on the junction detection dataset
# Each tuple is (dataset, trainer)
NNUNET_JUNCTION_MODELS: list[tuple[str, str]] = [
    # ("Dataset001_Segmentation_v1", "nnUNetTrainerClDiceLoss"),
]
