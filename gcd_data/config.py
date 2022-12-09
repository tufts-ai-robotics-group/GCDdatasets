import importlib.resources
from pathlib import Path

# dataset roots
with importlib.resources.path("gcd_data", "data") as data_root_raw:
    data_root = Path(data_root_raw)
cifar_10_root = data_root / "cifar10"
cifar_100_root = data_root / "cifar100"
cub_root = data_root / "cub"
aircraft_root = data_root / "fgvc-aircraft-2013b"
car_root = data_root / "cars"
herbarium_dataroot = data_root / "herbarium_19"
imagenet_root = data_root / "ImageNet"

# OSR Split dir
osr_split_dir = data_root / "ssb_splits"
