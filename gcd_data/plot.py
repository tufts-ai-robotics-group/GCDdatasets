from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import argparse

from gcd_data.get_datasets import get_class_splits, get_imbalanced_datasets, get_datasets
from gcd_data.data_utils import get_targets


def plot_imbalance(datasets, dataset_name):
    """Plot class imbalance on labeled and unlabeled datasets
    Args:
        datasets (dict): Dictionary of datasets returned by get_datasets
                         It should contain the following keys:
                            "train_labeled", "train_unlabeled", "test", "val"
        dataset_name (str): Name of dataset
    Returns:
        plt.Figure: Figure with stacked bar plots of class counts
                    for labeled and unlabeled datasets
    """
    # Get class counts for labeled and unlabeled training sets
    labeled_targets = get_targets(datasets["train_labeled"], dataset_name)
    unlabeled_targets = get_targets(datasets["train_unlabeled"], dataset_name)
    total_classes = len(np.unique(np.concatenate([labeled_targets, unlabeled_targets])))
    train_labeled_counts = np.bincount(labeled_targets, minlength=total_classes+1)
    train_unlabeled_counts = np.bincount(unlabeled_targets, minlength=total_classes+1)

    # Make figure and axes
    #  plots class balances for each dataset, ordered from largest to smallest
    #  and separated between labeled and unlabeled classes
    fig, ax = plt.subplots(1, 1, figsize=(20, 4))
    # ax.set_title("Labeled Training Set")
    # axes[1].set_title("Unlabeled Training Set")

    # Get class counts for pure unlabeled training set
    #  (i.e. no labeled classes)
    unlabeled_classes = np.setdiff1d(np.unique(unlabeled_targets),
                                     np.unique(labeled_targets))
    train_unlabeled_counts_pure = train_unlabeled_counts[unlabeled_classes]

    # Get class counts for labeled training set, which also includes unlabeled classes
    labeled_classes = np.unique(labeled_targets)

    # Plot class counts
    # Stacked bar plot for labeled training set
    ax.bar(np.arange(len(labeled_classes)), train_labeled_counts[labeled_classes],
           label="Labeled", alpha=.7)
    ax.bar(np.arange(len(labeled_classes)), train_unlabeled_counts[labeled_classes],
           bottom=train_labeled_counts[labeled_classes], label="Unlabeled",
           color='darkorange', alpha=.7)

    # Bar plot for unlabeled training set
    ax.bar(np.arange(len(labeled_classes),
                     len(labeled_classes) + len(unlabeled_classes)),
           train_unlabeled_counts_pure, color='darkorange', alpha=.7)

    # Set xticks
    all_classes = np.concatenate((labeled_classes, unlabeled_classes))
    ax.set_xticks(np.arange(len(labeled_classes) + len(unlabeled_classes)))
    ax.set_xticklabels(all_classes)

    ax.legend()
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    for dataset_name in ['cifar10', 'cifar100', 'cub', 'herbarium_19',
                         'novelcraft', 'scars', 'aircraft']:
        args = get_class_splits(argparse.Namespace(
            dataset_name=dataset_name, prop_train_labels=.5))

        # Get balanced datasets
        balanced_datasets = get_datasets(args.dataset_name, None, None, args)[3]
        fig = plot_imbalance(balanced_datasets, args.dataset_name)
        fig_dir = Path(f"figures/{args.dataset_name}")
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / "balanced.png")
        plt.close(fig)

        # Get imbalanced datasets
        for imbalance_ratio in [2, 10]:
            args.imbalance_ratio = imbalance_ratio
            # step imbalance
            args.imbalance_method = "step"
            for prop_minority_class in [0.2, 0.9]:
                args.prop_minority_class = prop_minority_class

                imbalanced_datasets = get_imbalanced_datasets(
                    args.dataset_name, None, None, args)[3]

                # Plot class imbalance
                fig = plot_imbalance(imbalanced_datasets, args.dataset_name)
                fig.savefig(
                    fig_dir / f"step-imbalance{imbalance_ratio}-minority{prop_minority_class}.png")
                plt.close(fig)

            # linear imbalance
            args.imbalance_method = "linear"

            imbalanced_datasets = get_imbalanced_datasets(args.dataset_name, None,
                                                          None, args)[3]
            fig = plot_imbalance(imbalanced_datasets, args.dataset_name)
            fig.savefig(fig_dir / f"linear-imbalance{imbalance_ratio}.png")
            plt.close(fig)
