import re
from pathlib import Path


def get_base_images(imgs_dir: Path) -> list[str]:
    if not imgs_dir.exists():
        raise ValueError(f"Images directory {imgs_dir} does not exist.")

    base_image_paths = set()
    pattern = re.compile(
        r'(\d{8}_.+?_tileset_\d+_tile_\d+_\d+)'
    )

    for img_path in imgs_dir.glob("*.png"):
        match = pattern.search(img_path.stem)
        if match and match not in base_image_paths:
            base_image_paths.add(match.group(1))

    return list(base_image_paths)
