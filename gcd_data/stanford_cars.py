import numpy as np
from pathlib import Path
from copy import deepcopy
from scipy import io as mat_io

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from gcd_data.data_utils import subsample_instances
from gcd_data.config import car_root


class CarsDataset(Dataset):
    """
        Cars Dataset
    """
    urls = [
        "http://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
        "http://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
        "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
        "http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat"
    ]

    def __init__(self, train=True, limit=0, data_dir=car_root, transform=None):
        data_dir = Path(data_dir)
        metas = data_dir / 'devkit/cars_train_annos.mat' if train else \
            data_dir / 'devkit/cars_test_annos_withlabels.mat'
        data_dir = data_dir / 'cars_train/' if train else data_dir / 'cars_test/'

        self.loader = default_loader
        self.data_dir = data_dir
        self.data = []
        self.target = []
        self.train = train

        self.transform = transform

        self.download()

        labels_meta = mat_io.loadmat(metas)

        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx > limit:
                    break

            # self.data.append(img_resized)
            self.data.append(data_dir / img_[5][0])
            # if self.mode == 'train':
            self.target.append(img_[4][0][0])

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

    def __getitem__(self, idx):

        image = self.loader(self.data[idx])
        target = self.target[idx] - 1

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        return image, target, idx

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self.data_dir.exists()

    def download(self):
        """Download the StanfordCars data if it doesn't exist already."""
        from torchvision.datasets.utils import download_url
        import tarfile

        if self._check_exists():
            return

        parent_dir = self.data_dir.parent
        parent_dir.mkdir(exist_ok=True)

        for url in self.urls:
            if url[-4:] == ".tgz":
                tar_filename = "temp.tar.gz"
                download_url(url, parent_dir, tar_filename)
                tar_path = parent_dir / tar_filename
                tar = tarfile.open(tar_path)
                tar.extractall(parent_dir)
                tar.close()
                tar_path.unlink()
            elif "cars_test_annos_withlabels.mat" in url:
                download_url(url, parent_dir / "devkit", "cars_test_annos_withlabels.mat")
            else:
                raise Exception(f"Unexpected URL in Stanford Cars: {url}")


def subsample_dataset(dataset, idxs):

    dataset.data = np.array(dataset.data)[idxs].tolist()
    dataset.target = np.array(dataset.target)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    # SCars classes are indexed 1 --> 196 instead of 0 --> 195
    include_classes_cars = np.array(include_classes) + 1
    cls_idxs = [x for x, t in enumerate(dataset.target) if t in include_classes_cars]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.target)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.target == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_scars_datasets(train_transform, test_transform, train_classes=range(160),
                       prop_train_labels=0.8, split_train_val=False, seed=0):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CarsDataset(data_dir=car_root, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(
        deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(
        train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(
        deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = CarsDataset(data_dir=car_root, transform=test_transform, train=False)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else \
        train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


if __name__ == '__main__':

    x = get_scars_datasets(None, None, train_classes=range(
        98), prop_train_labels=0.5, split_train_val=False)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].target))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].target))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')
