import argparse
import pytest

import numpy as np
from PIL.Image import Image

from gcd_data.get_datasets import get_datasets, get_class_splits


@pytest.fixture
def dataset_dict(dataset_name):
    args = get_class_splits(argparse.Namespace(
        dataset_name=dataset_name, prop_train_labels=.5))
    datasets = get_datasets(dataset_name, None, None, args)[3]
    return datasets


@pytest.mark.parametrize(
    "dataset_name,dataset_dict,lens,class_counts",
    [
        ("cifar10", "cifar10", [12500, 37500], [5, 10]),
        ("cifar100", "cifar100", [20000, 30000], [80, 100]),
        ("cub", "cub", [1498, 4496], [100, 200]),
        ("aircraft", "aircraft", [1666, 5001], [50, 100]),
        ("herbarium_19", "herbarium_19", [8869, 25356], [341, 683]),
        ("scars", "scars", [2000, 6144], [98, 196]),
        ("novelcraft", "novelcraft", [7037, 1205], [5, 10])
    ],
    indirect=["dataset_dict"]
)
class TestDatasets:
    def test_lens(self, dataset_name, dataset_dict, lens, class_counts):
        assert lens[0] == len(dataset_dict["train_labeled"])
        assert lens[1] == len(dataset_dict["train_unlabeled"])

    def test_class_count(self, dataset_name, dataset_dict, lens, class_counts):
        if dataset_name in ["cifar10", "cifar100", "herbarium_19", "novelcraft"]:
            assert class_counts[0] == len(set(dataset_dict["train_labeled"].targets))
            assert class_counts[1] == len(set(dataset_dict["train_unlabeled"].targets))
        elif dataset_name == "scars":
            assert class_counts[0] == len(set(dataset_dict["train_labeled"].target))
            assert class_counts[1] == len(set(dataset_dict["train_unlabeled"].target))
        elif dataset_name == "cub":
            assert class_counts[0] == len(set(dataset_dict["train_labeled"].data["target"]))
            assert class_counts[1] == len(set(dataset_dict["train_unlabeled"].data["target"]))
        elif dataset_name == "aircraft":
            assert class_counts[0] == len(set(
                [dataset_dict["train_labeled"].samples[i][1]
                 for i in range(len(dataset_dict["train_labeled"].samples))]))
            assert class_counts[1] == len(set(
                [dataset_dict["train_unlabeled"].samples[i][1]
                 for i in range(len(dataset_dict["train_unlabeled"].samples))]))
        else:
            assert False

    def test_contents(self, dataset_name, dataset_dict, lens, class_counts):
        for dataset in dataset_dict.values():
            if dataset is not None:
                output = dataset[0]
                assert len(output) == 3
                image, target, uq_id = output
                assert type(image) == Image
                assert type(target) == int
                assert uq_id.dtype == np.int32 and uq_id.ndim == 0
