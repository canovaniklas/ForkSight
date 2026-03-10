import sys
import os
from typing import Tuple
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from huggingface_hub import login


sys.path.append(os.path.abspath(".."))
from Environment.env_utils import load_segmentation_env

print(sys.executable, sys.version_info)
assert sys.version_info.major == 3 and sys.version_info.minor == 12, "Python version does not match the expected version."

login()

import sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

load_segmentation_env()

RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
HIGHRES_IMG_DIR_NAME = os.getenv("HIGHRES_IMG_DIR_NAME", "images_4096")
SAM3_OUTPUT_DIR_NAME = os.getenv("SAM3_OUTPUT_DIR_NAME", "sam3_outputs")


def zeroshot_segmentation_textprompt(image_path, prompt="lines", model=None, resize_dims=(1024, 1024)):
    image = Image.open(image_path)
    img_orig_size = image.size
    image = image.resize(resize_dims)

    processor = Sam3Processor(model, confidence_threshold=0.5)
    inference_state = processor.set_image(image)

    processor.reset_all_prompts(inference_state)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    masks = output["masks"]

    return masks.cpu(), img_orig_size


def export_np_mask(mask: np.ndarray, filename: str, img_orig_size: Tuple[int, int]):
    sam_output_dir = Path(RAW_DATA_DIR) / SAM3_OUTPUT_DIR_NAME
    mask_2d = mask.squeeze().astype(np.uint8) * 255
    img_mask = Image.fromarray(mask_2d, mode="L")
    img_mask = img_mask.resize(img_orig_size, resample=Image.NEAREST)
    img_mask.save(sam_output_dir / filename, format="PNG")


def main():
    # turn on tfloat32 for Ampere GPUs
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    bpe_path = f"/home/jhehli/data/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"SAM3 total parameters: {total_params}")
    print(f"SAM3 trainable parameters: {trainable_params} \n")

    device = torch.device("cuda:0")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    print(f"GPU total memory: {total_memory / (1024**3):.2f} GB")

    resize_dims = (1024, 1024)

    images_dir = Path(RAW_DATA_DIR) / HIGHRES_IMG_DIR_NAME

    masks = {}

    for img_path in images_dir.glob("*.png"):
        img_name = img_path.name
        print(img_name)

        output_masks, img_orig_size = zeroshot_segmentation_textprompt(
            str(img_path), prompt="lines", model=model, resize_dims=resize_dims)
        masks[img_name] = (output_masks.any(
            dim=0).numpy().astype(np.uint8), img_orig_size)

    for filename, (mask, img_orig_size) in masks.items():
        export_np_mask(mask, filename, img_orig_size)


if __name__ == "__main__":
    main()
