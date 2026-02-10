import re
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as F
import torch
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple


def get_base_images(imgs_dir: Path, exclude_soi_images: bool = True) -> list[str]:
    if not imgs_dir.exists():
        raise ValueError(f"Images directory {imgs_dir} does not exist.")

    base_image_paths = set()
    pattern = re.compile(
        r'(\d{8}_.+?_tileset_\d+_tile_\d+_\d+(?:_soi)?)'
    )

    for img_path in imgs_dir.glob("*.png"):
        if exclude_soi_images and "_soi_" in img_path.stem:
            continue
        match = pattern.search(img_path.stem)
        if match and match not in base_image_paths:
            base_image_paths.add(match.group(1))

    return list(base_image_paths)


def create_patches_from_img(input_image_path: Path, patch_size: int = 1024) -> torch.Tensor:
    img = Image.open(input_image_path)
    img = F.to_tensor(img).unsqueeze(0)

    patches = img.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size)
    _, C, _, _, H, W = patches.shape

    return patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, H, W)

def parse_junction_annotations_xml(xml_path: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Parse CVAT 1.1 XML file and extract point annotations per image.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    points_per_image = {}

    for image_elem in root.findall('.//image'):
        image_name = image_elem.get('name')
        points = []

        for points_elem in image_elem.findall('.//points'):
            points_str = points_elem.get('points')
            if points_str:
                coords = points_str.strip().split(',')
                if len(coords) == 2:
                    x, y = float(coords[0]), float(coords[1])
                    points.append((x, y))

        if points:
            points_per_image[image_name] = points

    return points_per_image
