import os

import torchvision
import numpy as np
from copy import deepcopy

from gcd_data.data_utils import subsample_instances
from gcd_data.config import herbarium_dataroot


class HerbariumDataset19(torchvision.datasets.ImageFolder):
    urls = [
        "https://storage.googleapis.com/nybg/herbarium-2019-fgvc6/small-train.tar.gz",
        "https://storage.googleapis.com/nybg/herbarium-2019-fgvc6/small-validation.tar.gz"
    ]

    def __init__(self, *args, **kwargs):
        self.root = kwargs["root"]
        self.download()

        # Process metadata json for training images into a DataFrame
        super().__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, idx):

        img, label = super().__getitem__(idx)
        uq_idx = self.uq_idxs[idx]

        return img, label, uq_idx

    def _check_exists(self):
        return os.path.exists(self.root)

    def download(self):
        """Download the herbarium_19 data if it doesn't exist already."""
        from torchvision.datasets.utils import download_url
        import tarfile

        if self._check_exists():
            return

        parent_dir = os.path.join(self.root, os.path.pardir)
        os.makedirs(parent_dir, exist_ok=True)

        for url in self.urls:
            tar_filename = "temp.tar.gz"
            download_url(url, parent_dir, tar_filename)
            tar_path = os.path.join(parent_dir, tar_filename)
            tar = tarfile.open(tar_path)
            tar.extractall(parent_dir)
            tar.close()
            os.remove(tar_path)


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.samples = np.array(dataset.samples)[mask].tolist()
    dataset.targets = np.array(dataset.targets)[mask].tolist()

    dataset.uq_idxs = dataset.uq_idxs[mask]

    dataset.samples = [[x[0], int(x[1])] for x in dataset.samples]
    dataset.targets = [int(x) for x in dataset.targets]

    return dataset


def subsample_classes(dataset, include_classes=range(250)):

    cls_idxs = [x for x, l in enumerate(dataset.targets) if l in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_instances_per_class=5):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        # Have a balanced test set
        v_ = np.random.choice(cls_idxs, replace=False, size=(val_instances_per_class,))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_herbarium_datasets(train_transform, test_transform, train_classes=range(500),
                           prop_train_labels=0.8, seed=0, split_train_val=False):

    np.random.seed(seed)

    # Init entire training set
    train_dataset = HerbariumDataset19(transform=train_transform,
                                       root=os.path.join(herbarium_dataroot, 'small-train'))

    # Get labeled training set which has subsampled classes, then subsample some indices from that
    # TODO: Subsampling unlabeled set in uniform random fashion from training data,
    # will contain many instances of dominant class
    train_dataset_labeled = subsample_classes(
        deepcopy(train_dataset), include_classes=train_classes)
    subsample_indices = subsample_instances(
        train_dataset_labeled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labeled = subsample_dataset(train_dataset_labeled, subsample_indices)

    # Split into training and validation sets
    if split_train_val:

        train_idxs, val_idxs = get_train_val_indices(train_dataset_labeled,
                                                     val_instances_per_class=5)
        train_dataset_labeled_split = subsample_dataset(
            deepcopy(train_dataset_labeled), train_idxs)
        val_dataset_labeled_split = subsample_dataset(deepcopy(train_dataset_labeled), val_idxs)
        val_dataset_labeled_split.transform = test_transform

    else:

        train_dataset_labeled_split, val_dataset_labeled_split = None, None

    # Get unlabeled data
    unlabeled_indices = set(train_dataset.uq_idxs) - set(train_dataset_labeled.uq_idxs)
    train_dataset_unlabeled = subsample_dataset(
        deepcopy(train_dataset), np.array(list(unlabeled_indices)))

    # Get test dataset
    test_dataset = HerbariumDataset19(transform=test_transform,
                                      root=os.path.join(herbarium_dataroot, 'small-validation'))

    # Transform dict
    unlabeled_classes = list(set(train_dataset.targets) - set(train_classes))
    target_xform_dict = {}
    for i, k in enumerate(list(train_classes) + unlabeled_classes):
        target_xform_dict[k] = i

    test_dataset.target_transform = lambda x: target_xform_dict[x]
    train_dataset_unlabeled.target_transform = lambda x: target_xform_dict[x]

    # Either split train into train and val or use test set as val
    train_dataset_labeled = train_dataset_labeled_split if split_train_val else \
        train_dataset_labeled
    val_dataset_labeled = val_dataset_labeled_split if split_train_val else None

    all_datasets = {
        'train_labeled': train_dataset_labeled,
        'train_unlabeled': train_dataset_unlabeled,
        'val': val_dataset_labeled,
        'test': test_dataset,
    }

    return all_datasets


if __name__ == '__main__':

    np.random.seed(0)
    train_classes = np.random.choice(range(683,), size=(int(683 / 2)), replace=False)

    x = get_herbarium_datasets(None, None, train_classes=train_classes,
                               prop_train_labels=0.5)

    assert set(x['train_unlabeled'].targets) == set(range(683))

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labeled and unlabeled overlap...')
    print(set.intersection(set(x['train_labeled'].uq_idxs), set(x['train_unlabeled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labeled'].uq_idxs)) + len(set(x['train_unlabeled'].uq_idxs)))
    print('Printing number of labeled classes...')
    print(len(set(x['train_labeled'].targets)))
    print('Printing total number of classes...')
    print(len(set(x['train_unlabeled'].targets)))

    print(f'Num labeled Classes: {len(set(x["train_labeled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabeled"].targets))}')
    print(f'Len labeled set: {len(x["train_labeled"])}')
    print(f'Len unlabeled set: {len(x["train_unlabeled"])}')
