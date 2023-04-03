from pathlib import Path

import argparse
import pytest

import numpy as np
import pickle

from gcd_data.get_datasets import get_class_splits, get_imbalanced_datasets, get_uq_idx


@pytest.mark.parametrize(
    "dataset_name,imbalance_method,imbalance_ratio,prop_minority_class,seed",
    [
        ("cifar10", "linear", 2, 0.5, 0),
        ("cifar10", "linear", 2, 0.9, 0),
        ("cifar10", "linear", 10, 0.9, 0),
        ("cifar10", "step", 2, 0.5, 0),
        ("cifar10", "step", 2, 0.9, 0),
        ("cifar10", "step", 10, 0.9, 0),
        ("cifar10", "step", 10, 0.9, 123),
        ("cifar100", "linear", 2, 0.5, 0),
        ("cub", "linear", 2, 0.5, 0),
        ("aircraft", "linear", 2, 0.5, 0),
        ("herbarium_19", "linear", 2, 0.5, 0),
        ("scars", "linear", 2, 0.5, 0),
        ("novelcraft", "linear", 2, 0.5, 0),
    ]
)
class TestImbalance:
    def test_imbalance_consistency(self, dataset_name, imbalance_method, imbalance_ratio,
                                   prop_minority_class, seed):
        args = get_class_splits(argparse.Namespace(
            dataset_name=dataset_name, prop_train_labels=.5,
            imbalance_method=imbalance_method, imbalance_ratio=imbalance_ratio,
            prop_minority_class=prop_minority_class, seed=seed))
        datasets = get_imbalanced_datasets(dataset_name, None, None, args)[3]

        imbalance_params = (f'{imbalance_method}'
                            f'_ratio{imbalance_ratio}'
                            f'_minority{prop_minority_class}'
                            f'_seed{seed}')

        indices_dir = Path(f"gcd_data/data/imbalanced/{dataset_name}/{imbalance_params}")

        for split in ['train_labeled', 'train_unlabeled']:
            indices = get_uq_idx(datasets[split])

            with open(indices_dir / f"{split}.pkl", "rb") as f:
                expected_indices = pickle.load(f)
                assert np.array_equal(indices, expected_indices)
