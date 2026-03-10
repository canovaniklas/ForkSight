import os
from dotenv import load_dotenv
from pathlib import Path


def find_repo_root() -> Path:
    start = Path(__file__).resolve()
    for parent in [start] + list(start.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("No repo root ('.git' folder) found")


def load_segmentation_env():
    repo_root = find_repo_root()
    env_path = repo_root / "Environment" / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        print(f"loaded environment variables from: {env_path}")
    else:
        print(f".env file not found: {env_path} — using defaults")


def load_as_tuple(var: str, default=None, dtype=int) -> tuple:
    val = os.getenv(var, default)
    if val is None or val.strip() == "":
        return None
    parts = val.strip().split(',')
    return tuple(dtype(p.strip()) for p in parts)


def load_as(var: str, dtype, default=None):
    val = os.getenv(var, default)
    if val is None:
        return None
    return dtype(val)


def load_as_bool(var: str, default=False) -> bool:
    val = os.getenv(var, str(default)).lower().strip()
    if val == 'true':
        return True
    elif val == 'false':
        return False
    else:
        raise ValueError(f"Cannot convert {val} to bool")
