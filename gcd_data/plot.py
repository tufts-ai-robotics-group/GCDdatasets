from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import argparse

from gcd_data.get_datasets import get_class_splits, get_imbalanced_datasets, get_datasets

fig_size = {
    'cifar10': 10,
    'cifar100': 20,
    'herbarium_19': 100,
    'cub': 30,
    'aircraft': 20,
    'scars': 40,
    'novelcraft': 10,
}


def plot_imbalance(datasets, dataset_name):
    """Plot class imbalance on labeled and unlabeled datasets of the train set in log scale
       The normal classes are plotted in the order of the number of labaled instances
       The novel classes are plotted in the order of the number of the total instances
       Assumes that the labels are integers, and the normal classes are indexed smaller than
       the novel calsses
    Args:
        datasets (dict): Dictionary of datasets returned by get_datasets
                         It should contain the following keys:
                            "train_labeled", "train_unlabeled", "test", "val"
    Returns:
        plt.Figure: Figure with stacked bar plots of class counts
                    for labeled and unlabeled datasets in log scale
    """
    # Get class counts for labeled and unlabeled training sets
    labeled_targets = np.array([datasets['train_labeled'][i][1]
                               for i in range(len(datasets['train_labeled']))])
    unlabeled_targets = np.array([datasets['train_unlabeled'][i][1]
                                 for i in range(len(datasets['train_unlabeled']))])
    total_cls = len(np.unique(np.concatenate([labeled_targets, unlabeled_targets])))
    labeled_counts = np.bincount(labeled_targets, minlength=total_cls+1)
    unlabeled_counts = np.bincount(unlabeled_targets, minlength=total_cls+1)

    fig, ax = plt.subplots(1, 1, figsize=(fig_size[dataset_name], 4))

    if dataset_name in ['herbarium_19']:
        ax.set_yscale('log')

    # Get class counts for novel cls
    normal_cls = np.unique(labeled_targets)
    novel_cls = np.setdiff1d(np.unique(unlabeled_targets), normal_cls)
    novel_counts = unlabeled_counts[novel_cls]

    # Order classes by class counts
    normal_cls = normal_cls[np.argsort(labeled_counts[normal_cls])[::-1]]
    labeled_counts = labeled_counts[normal_cls]
    novel_cls = novel_cls[np.argsort(novel_counts)[::-1]]
    novel_counts = unlabeled_counts[novel_cls]

    # Plot class counts side by side for labeled and unlabeled training sets of the normal classes
    # Side by side bar plot for labeled training set
    ax.bar(np.arange(len(normal_cls))-0.2, labeled_counts, label='Labeled', alpha=.7, width=.4)
    # Side by side bar plot for unlabeled training set
    ax.bar(np.arange(len(normal_cls))+0.2, unlabeled_counts[normal_cls],
           color='darkorange', width=.4, label='Unlabeled', alpha=.7)

    # Bar plot for unlabeled training set
    ax.bar(np.arange(len(normal_cls),
                     len(normal_cls) + len(novel_cls)),
           novel_counts, color='darkorange', alpha=.7, width=.4)

    # Set xticks
    ax.set_xticks(np.arange(total_cls), labels=np.concatenate([normal_cls, novel_cls]))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    ax.legend()
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    for dataset_name in ['cifar10', 'cifar100', 'herbarium_19', 'cub', 'aircraft', 'scars',
                         'novelcraft']:
        print(f"Plotting class imbalance for {dataset_name}...")
        args = get_class_splits(argparse.Namespace(
            dataset_name=dataset_name, prop_train_labels=.5))

        # Get balanced datasets
        original_datasets = get_datasets(args.dataset_name, None, None, args)[3]
        fig = plot_imbalance(original_datasets, dataset_name=dataset_name)
        fig_dir = Path(f"figures/{args.dataset_name}")
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / "original.png")
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
                fig = plot_imbalance(imbalanced_datasets, dataset_name=dataset_name)
                fig.savefig(
                    fig_dir / f"step-imbalance{imbalance_ratio}-minority{prop_minority_class}.png")
                plt.close(fig)

            # linear imbalance
            args.imbalance_method = "linear"

            imbalanced_datasets = get_imbalanced_datasets(args.dataset_name, None,
                                                          None, args)[3]
            fig = plot_imbalance(imbalanced_datasets, dataset_name=dataset_name)
            fig.savefig(fig_dir / f"linear-imbalance{imbalance_ratio}.png")
            plt.close(fig)
