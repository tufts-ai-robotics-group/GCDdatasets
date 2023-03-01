from polycraft_nov_data.dataset import NovelCraft
from polycraft_nov_data.novelcraft_const import SplitEnum


def get_novelcraft_datasets(train_transform, test_transform, train_classes=None,
                            prop_train_labels=None, split_train_val=None, seed=None,
                            download=None):
    train_dataset_labeled = NovelCraft(SplitEnum.TRAIN, train_transform)
    train_dataset_unlabeled = NovelCraft(SplitEnum.VALID, train_transform)
    test_dataset = NovelCraft(SplitEnum.TEST, test_transform)
    all_datasets = {
        'train_labeled': train_dataset_labeled,
        'train_unlabeled': train_dataset_unlabeled,
        'val': None,
        'test': test_dataset,
    }
    return all_datasets
