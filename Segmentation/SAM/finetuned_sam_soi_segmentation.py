import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import wandb
import torchvision.transforms as transforms

from Segmentation.SAM.sam_lora_util import EVALUATED_TAG, get_batched_input_list, get_params_from_artifact, initialize_sam_lora_with_params
from Environment.env_utils import load_segmentation_env, load_as

load_segmentation_env()

SEED = load_as("SEED", int, 42)

RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
HIGHRES_IMG_DIR_NAME = os.getenv("HIGHRES_IMG_DIR_NAME", "images_4096")
FINETUNED_SAM_OUTPUT_DIR_NAME = os.getenv(
    "FINETUNED_SAM_OUTPUT_DIR_NAME", "finetuned_sam_output")

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "EM_IMCR_BIOVSION")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "ForkSight-SAM")


def init_finetuned_sam_model(device: torch.device):
    runs = [run for run in list(wandb.Api().runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}")) if EVALUATED_TAG in run.tags and run.state == "finished"]
    for i, run in enumerate(runs):
        print(f"({i}) {run.name}")
    run_idx = int(input("Enter the index of the run to evaluate: "))
    run = runs[run_idx]
    print(f"\nSelected run: {run.name} ({run.id})")

    model_param_artifacts = [a for a in list(
        run.logged_artifacts()) if a.type == "model"]
    for i, artifact in enumerate(model_param_artifacts):
        print(f"({i}) Artifact: {artifact.name}")
    artifact_idx = int(input("Enter the index of the artifact to evaluate: "))
    artifact = model_param_artifacts[artifact_idx]

    params, _ = get_params_from_artifact(artifact, device)
    model = initialize_sam_lora_with_params(run.config, params, device)
    model.eval()

    return model


def load_transform_image(path: Path):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t)])
    img = Image.open(path)
    return transform(img)


def export_np_mask(mask: np.ndarray, filename: str):
    finetuned_sam_output_dir = Path(
        RAW_DATA_DIR) / FINETUNED_SAM_OUTPUT_DIR_NAME
    mask_2d = mask.squeeze().astype(np.uint8) * 255
    img_mask = Image.fromarray(mask_2d, mode="L")
    img_mask = img_mask.resize((1024, 1024), resample=Image.NEAREST)
    img_mask.save(finetuned_sam_output_dir / filename, format="PNG")


def main():
    device = torch.device("cuda")
    model = init_finetuned_sam_model(device)

    images_dir = Path(RAW_DATA_DIR) / HIGHRES_IMG_DIR_NAME

    masks = {}
    for img_path in images_dir.glob("*_soi.png"):
        img_name = img_path.name
        print(img_name)

        img = load_transform_image(img_path)
        input_list = get_batched_input_list(img.unsqueeze(0).to(device))
        output = model(batched_input=input_list, multimask_output=False)
        output_mask = output[0]['masks'].squeeze(0).detach().cpu()
        masks[img_name] = output_mask.any(dim=0).numpy().astype(np.uint8)

    for filename, mask in masks.items():
        export_np_mask(mask, filename)


if __name__ == "__main__":
    main()
