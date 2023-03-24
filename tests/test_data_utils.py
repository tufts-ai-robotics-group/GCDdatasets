import argparse
import pytest

import torch

from gcd_data.data_utils import WeightedConsistentSampler, MergedDatasetSampler
from gcd_data.get_datasets import get_datasets, get_class_splits


@pytest.fixture
def merged_dataset(dataset_name):
    args = get_class_splits(argparse.Namespace(
        dataset_name=dataset_name, prop_train_labels=.5))
    return get_datasets(dataset_name, None, None, args)[0]


@pytest.mark.parametrize(
    "dataset_name, merged_dataset",
    [
        ("cifar10", "cifar10"),
        ("cifar100", "cifar100"),
        ("cub", "cub"),
        ("aircraft", "aircraft"),
        ("herbarium_19", "herbarium_19"),
        ("scars", "scars"),
        ("novelcraft", "novelcraft"),
    ],
    indirect=["merged_dataset"]
)
class TestMergedSampler:
    def test_merged_sampler(self, dataset_name, merged_dataset):
        sampler = MergedDatasetSampler(merged_dataset)
        sampler_len = 0
        for sample_index in sampler:
            # check that sampler alternates between labeled and unlabeled
            if sampler_len % 2 == 0:
                assert sample_index < len(merged_dataset.labeled_dataset) and sample_index >= 0
            else:
                assert sample_index >= len(merged_dataset.labeled_dataset) and \
                    sample_index < len(merged_dataset)
            sampler_len += 1
        # check sampler length
        assert sampler.epoch_size == sampler_len


def test_weighted_sampler():
    sampler = WeightedConsistentSampler(([1] * 100) + ([2] * 100), 200)
    samples = torch.Tensor([i for i in sampler])
    for i in range(2):
        assert torch.allclose(samples, torch.Tensor([i for i in sampler]))
