import os
import random
import sys

from sam_lora import SamLoRA
from segment_anything import sam_model_registry
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image


class SegmentationDataset(Dataset):
    def __init__(self, root_path: Path):
        self.image_paths = (root_path / "Images").rglob("*.png")

    def _get_mask_path(self, image_path: Path) -> Path:
        return image_path.parent.parent / "Masks" / image_path.name

    def _get_resize_transform(self, path: Path, normalize: bool = True) -> torch.Tensor:
        transform = transforms.Compose([transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
                                        transforms.ToTensor() if normalize else transforms.PILToTensor(),
                                        transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t)])
        img = Image.open(path)
        return transform(img)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self._get_mask_path(image_path)

        image = self._get_resize_transform(image_path)
        mask = self._get_resize_transform(mask_path, normalize=False)

        return {"image": image, "original_size": image.size()[1:]}, mask


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def init_model(device: torch.device) -> SamLoRA:
    print(
        f"using python: {sys.executable}, {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n")

    sam_checkpoint = "/data/jhehli/model_checkpoints/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)

    print(
        f"SAM model loaded on {sam.device}, with {sum(p.numel() for p in sam.parameters() if p.requires_grad)} trainable parameters")

    sam_lora = SamLoRA(sam, r=4)
    sam_lora.to(device)
    print(
        f"SAM model with LoRA fine-tuning initialized, on {sam_lora.device}, with {sum(p.numel() for p in sam.parameters() if p.requires_grad)} trainable parameters")
    # print(sam_lora)

    return sam_lora


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_lora = init_model(device)

    lr = 0.005
    num_classes = 1  # binary segmentation (DNA vs background)
    batch_size = 4
    max_epochs = 100

    optimizer = torch.optim.AdamW(params=filter(
        lambda p: p.requires_grad, sam_lora.parameters()), lr=lr)

    for epoch in range(max_epochs):
        sam_lora.train()


if __name__ == "__main__":
    seed_everything(42)
    train()
