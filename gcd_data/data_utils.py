import itertools

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, Sampler


def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices


class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:

            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0

        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)


class IndexDataset(Dataset):
    def __init__(self, source_dataset) -> None:
        """Wrap a dataset and return its contents with indices appended

        Args:
            source_dataset (Dataset): Dataset to be wrapped
        """
        super().__init__()
        self.source_dataset = source_dataset

    def __getitem__(self, item):
        return (*self.source_dataset[item], item)

    def __len__(self):
        return len(self.source_dataset)


class WeightedConsistentSampler(WeightedRandomSampler):
    def __init__(self, weights, num_samples: int, replacement: bool = True) -> None:
        super().__init__(weights, num_samples, replacement, torch.Generator())

    def __iter__(self):
        # reset generator so initial seed is consistently used
        self.generator.manual_seed(self.generator.initial_seed())
        return super().__iter__()


class MergedDatasetSampler(Sampler):
    def __init__(self, merged_dataset: MergedDataset) -> None:
        labeled_len = len(merged_dataset.labelled_dataset)
        unlabelled_len = len(merged_dataset.unlabelled_dataset)
        # construct epoch that's twice the size of the larger set
        self.epoch_size = max(labeled_len, unlabelled_len) * 2
        # construct samplers, with only the smaller set having replacement
        self.labeled_sampler = WeightedConsistentSampler(
            torch.Tensor([1] * labeled_len),
            num_samples=self.epoch_size//2,
            replacement=labeled_len < unlabelled_len)
        self.unlabeled_sampler = WeightedConsistentSampler(
            torch.Tensor(([0] * labeled_len) + ([1] * unlabelled_len)),
            num_samples=self.epoch_size//2,
            replacement=labeled_len > unlabelled_len)

    def __iter__(self):
        # alternates between labeled and unlabeled
        return itertools.chain(*zip(self.labeled_sampler.__iter__(),
                                    self.unlabeled_sampler.__iter__()))
