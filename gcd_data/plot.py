from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import argparse

from gcd_data.get_datasets import get_datasets, get_class_splits


def plot_imbalance(datasets):
    """Plot class imbalance on labeled and unlabeled datasets

    Args:
        datasets (dict): Dictionary of datasets returned by get_datasets
                         It should contain the following keys:
                            "train_labeled", "train_unlabeled", "test", "val"
    Returns:
        plt.Figure: Figure with stacked bar plots of class counts
                    for labeled and unlabeled datasets

    """
    # Get class counts for labeled and unlabeled training sets
    train_labeled_counts = np.bincount(datasets["train_labeled"].targets)
    train_unlabeled_counts = np.bincount(datasets["train_unlabeled"].targets)

    # Make figure and axes
    #  plots class balances for each dataset, ordered from largest to smallest
    #  and separated between labeled and unlabeled classes
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # ax.set_title("Labeled Training Set")
    # axes[1].set_title("Unlabeled Training Set")

    # Get class counts for pure unlabeled training set
    #  (i.e. no labeled classes)
    unlabeled_classes = np.setdiff1d(np.unique(datasets["train_unlabeled"].targets),
                                     np.unique(datasets["train_labeled"].targets))
    unlabeled_classes_names = np.array(datasets["train_unlabeled"].classes)[unlabeled_classes]
    train_unlabeled_counts_pure = train_unlabeled_counts[unlabeled_classes]

    # Get class counts for labeled training set, which also includes unlabeled classes
    labeled_classes = np.unique(datasets["train_labeled"].targets)
    labeled_classes_names = np.array(datasets["train_labeled"].classes)[labeled_classes]
    train_labeled_counts_on_unlabeled = train_unlabeled_counts[labeled_classes]

    # Plot class counts
    # Stacked bar plot for labeled training set
    ax.bar(np.arange(len(train_labeled_counts)), train_labeled_counts,
           label="Labeled", alpha=.7)
    ax.bar(np.arange(len(train_labeled_counts)), train_labeled_counts_on_unlabeled,
           bottom=train_labeled_counts, label="Unlabeled", color='darkorange', alpha=.7)

    # Bar plot for unlabeled training set
    ax.bar(np.arange(len(train_labeled_counts),
                     len(train_labeled_counts) + len(train_unlabeled_counts_pure)),
           train_unlabeled_counts_pure, color='darkorange', alpha=.7)

    # Set xticks
    all_classes = np.concatenate((labeled_classes_names, unlabeled_classes_names))
    ax.set_xticks(np.arange(len(train_labeled_counts) + len(train_unlabeled_counts_pure)))
    ax.set_xticklabels(all_classes)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

    ax.legend()
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Get datasets
    args = get_class_splits(argparse.Namespace(
        dataset_name="cifar10", prop_train_labels=.5))
    datasets = get_datasets("cifar10", None, None, args)[3]

    # Plot class imbalance
    fig = plot_imbalance(datasets)
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / f"{args.dataset_name}_imbalance.png")
