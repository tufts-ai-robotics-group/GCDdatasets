from copy import deepcopy
from pathlib import Path
import pickle

import polycraft_nov_data.novelcraft_const as nc_const

from gcd_data.config import osr_split_dir
from gcd_data.data_utils import MergedDataset

from gcd_data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from gcd_data.herbarium_19 import get_herbarium_datasets
from gcd_data.novelcraft import get_novelcraft_datasets
from gcd_data.stanford_cars import get_scars_datasets
from gcd_data.cub import get_cub_datasets
from gcd_data.fgvc_aircraft import get_aircraft_datasets

get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'herbarium_19': get_herbarium_datasets,
    'cub': get_cub_datasets,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets,
    'novelcraft': get_novelcraft_datasets,
}


def get_datasets(dataset_name, train_transform, test_transform, args):
    """Shared interfrace for getting datasets, with targets reordered so:
    Normal classes: [0, ..., len(list(args.train_classes))]
    Novel classes: [len(list(args.train_classes)) + 1, ...,
    len(list(args.train_classes)) + len(list(args.unlabeled_classes))]

    Args:
        dataset_name (str): key for get_dataset_funcs
        train_transform (Transform): transform for training set
        test_transform (Transform): transform for test set
        args (Namespace): Output of get_class_splits with prop_train_labels set

    Raises:
        ValueError: dataset_name not key for get_dataset_funcs

    Returns:
        tuple: train_dataset: MergedDataset which concatenates labeled (normal) and
                              unlabeled (normal and novel)
               validation_dataset: Disjoint validation set with normal and novel data
                                   where only normal labels should be used,
               test_dataset: Unlabeled training set with test transform,
               datasets: dict returned by dataset specific get_dataset function
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                             train_classes=args.train_classes,
                             prop_train_labels=args.prop_train_labels,
                             split_train_val=False)
    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i

    def target_transform(x): return target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labeled and unlabeled classes) for training
    train_dataset = MergedDataset(labeled_dataset=deepcopy(datasets['train_labeled']),
                                  unlabeled_dataset=deepcopy(datasets['train_unlabeled']))

    test_dataset = datasets['test']
    unlabeled_train_examples_test = deepcopy(datasets['train_unlabeled'])
    unlabeled_train_examples_test.transform = test_transform

    return train_dataset, test_dataset, unlabeled_train_examples_test, datasets


def get_class_splits(args):

    # For FGVC datasets, by default use SSB splits
    if args.dataset_name in ('scars', 'cub', 'aircraft'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = True

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'cifar10':

        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)

    elif args.dataset_name == 'cifar100':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'herbarium_19':

        args.image_size = 224
        herb_path_splits = Path(osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']

    elif args.dataset_name == 'imagenet_100':

        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'scars':

        args.image_size = 224

        if use_ssb_splits:

            split_path = Path(osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + \
                open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)

    elif args.dataset_name == 'aircraft':

        args.image_size = 224
        if use_ssb_splits:

            split_path = Path(osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + \
                open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'cub':

        args.image_size = 224

        if use_ssb_splits:

            split_path = Path(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + \
                open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)

    elif args.dataset_name == 'novelcraft':

        args.image_size = nc_const.IMAGE_SHAPE[1]
        args.train_classes = nc_const.NORMAL_CLASSES
        args.unlabeled_classes = nc_const.NOVEL_VALID_CLASSES + nc_const.NOVEL_TEST_CLASSES

    else:

        raise NotImplementedError

    return args
