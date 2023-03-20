from polycraft_nov_data.dataset import NovelCraft
from polycraft_nov_data.novelcraft_const import SplitEnum

from gcd_data.data_utils import IndexDataset
from torch.utils.data import Subset


def get_novelcraft_datasets(train_transform, test_transform, train_classes=None,
                            prop_train_labels=None, split_train_val=None, seed=None,
                            download=None):
    train_dataset_labeled = NovelCraft(SplitEnum.TRAIN, train_transform)
    train_dataset_unlabeled = NovelCraft(SplitEnum.VALID, train_transform)
    test_dataset = NovelCraft(SplitEnum.TEST, test_transform)
    all_datasets = {
        'train_labeled': IndexDataset(train_dataset_labeled),
        'train_unlabeled': IndexDataset(train_dataset_unlabeled),
        'val': None,
        'test': IndexDataset(test_dataset),
    }
    return all_datasets


def subsample_dataset(dataset, idxs):

    return Subset(dataset, idxs)
