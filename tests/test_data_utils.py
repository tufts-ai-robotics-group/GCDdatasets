import argparse
import pytest

import torch

from gcd_data.data_utils import WeightedConsistentSampler, MergedDatasetSampler
from gcd_data.get_datasets import get_datasets, get_class_splits


@pytest.mark.parametrize(
    "dataset_name",
    [
        "cifar10",
        "cifar100",
        "cub",
        "aircraft",
        "herbarium_19",
        "scars",
    ],
)
class TestMergedSampler:
    def merged_dataset(self, dataset_name):
        args = get_class_splits(argparse.Namespace(
            dataset_name=dataset_name, use_ssb_splits=True, prop_train_labels=.5))
        return get_datasets(dataset_name, None, None, args)[0]

    def test_sampler(self, dataset_name):
        merged_dataset = self.merged_dataset(dataset_name)
        sampler = MergedDatasetSampler(merged_dataset)
        sampler_len = 0
        for sample_index in sampler:
            # check that sampler alternates between labeled and unlabeled
            if sampler_len % 2 == 0:
                assert sample_index < len(merged_dataset.labelled_dataset) and sample_index >= 0
            else:
                assert sample_index >= len(merged_dataset.labelled_dataset) and \
                    sample_index < len(merged_dataset)
            sampler_len += 1
        # check sampler length
        assert sampler.epoch_size == sampler_len


def test_sampler():
    sampler = WeightedConsistentSampler(([1] * 100) + ([2] * 100), 200)
    samples = torch.Tensor([i for i in sampler])
    for i in range(2):
        assert torch.allclose(samples, torch.Tensor([i for i in sampler]))
