from pathlib import Path

import argparse
import pytest

import numpy as np
import pickle

from gcd_data.get_datasets import get_class_splits, get_imbalanced_datasets


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

        indices_dir = Path(f"gcd_data/data/imbalanced/{dataset_name}")

        train_labeled_indices = np.array([datasets['train_labeled'][i][2]
                                          for i in range(len(datasets['train_labeled']))])
        train_unlabeled_indices = np.array([datasets['train_unlabeled'][i][2]
                                            for i in range(len(datasets['train_unlabeled']))])

        filename = ('train_labeled'
                    f'_{imbalance_method}'
                    f'_imbalance_ratio{imbalance_ratio}'
                    f'_prop_minority_class{prop_minority_class}'
                    f'_seed{seed}.pkl')

        with open(indices_dir / filename, "rb") as f:
            expected_train_labeled_indices = pickle.load(f)
            assert np.array_equal(train_labeled_indices, expected_train_labeled_indices)

        filename = ('train_unlabeled'
                    f'_{imbalance_method}'
                    f'_imbalance_ratio{imbalance_ratio}'
                    f'_prop_minority_class{prop_minority_class}'
                    f'_seed{seed}.pkl')
        with open(indices_dir / filename, "rb") as f:
            expected_train_unlabeled_indices = pickle.load(f)
            assert np.array_equal(train_unlabeled_indices, expected_train_unlabeled_indices)
