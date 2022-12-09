import argparse
import pytest

from gcd_data.get_datasets import get_datasets, get_class_splits


@pytest.mark.parametrize(
    "dataset_name,lens,class_counts",
    [
        ("cifar10", [12500, 37500], [5, 10]),
        ("cifar100", [20000, 30000], [80, 100]),
        ("cub", [1498, 4496], [100, 200]),
        ("aircraft", [1666, 5001], [50, 100]),
        ("herbarium_19", [8869, 25356], [341, 683]),
        ("scars", [2000, 6144], [98, 196]),
    ],
)
class TestDatasets:
    def dataset_dict(self, dataset_name):
        args = get_class_splits(argparse.Namespace(
            dataset_name=dataset_name, use_ssb_splits=True, prop_train_labels=.5))
        datasets = get_datasets(dataset_name, None, None, args)[3]
        return datasets

    def test_lens(self, dataset_name, lens, class_counts):
        dataset_dict = self.dataset_dict(dataset_name)
        assert lens[0] == len(dataset_dict["train_labelled"])
        assert lens[1] == len(dataset_dict["train_unlabelled"])

    def test_class_count(self, dataset_name, lens, class_counts):
        dataset_dict = self.dataset_dict(dataset_name)
        if dataset_name in ["cifar10", "cifar100", "herbarium_19"]:
            assert class_counts[0] == len(set(dataset_dict["train_labelled"].targets))
            assert class_counts[1] == len(set(dataset_dict["train_unlabelled"].targets))
        elif dataset_name == "scars":
            assert class_counts[0] == len(set(dataset_dict["train_labelled"].target))
            assert class_counts[1] == len(set(dataset_dict["train_unlabelled"].target))
        elif dataset_name == "cub":
            assert class_counts[0] == len(set(dataset_dict["train_labelled"].data["target"]))
            assert class_counts[1] == len(set(dataset_dict["train_unlabelled"].data["target"]))
        elif dataset_name == "aircraft":
            assert class_counts[0] == len(set(
                [dataset_dict["train_labelled"].samples[i][1]
                 for i in range(len(dataset_dict["train_labelled"].samples))]))
            assert class_counts[1] == len(set(
                [dataset_dict["train_unlabelled"].samples[i][1]
                 for i in range(len(dataset_dict["train_unlabelled"].samples))]))
        else:
            assert False
