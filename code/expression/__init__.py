
#from .losses import AgeLoss
#from .verification import FGNetVerification, CelebAVerification, RAFVerification, LAPVerification
from .verification import ExpressionVerification
from .models import SwinTransFER

from timm.data import Mixup
from timm.data import create_transform

import torch
import numpy as np
import torch.distributed as dist

from typing import Iterable
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info

from .datasets import RAFDBDataset, AffectNetDataset, FERPlusDataset
from .samplers import SubsetRandomSampler

from typing import Callable
import pandas as pd
import torchvision

def get_mixup_fn(config):

    mixup_fn = None
    mixup_active = config.AUG_MIXUP > 0 or config.AUG_CUTMIX > 0. or config.AUG_CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG_MIXUP, cutmix_alpha=config.AUG_CUTMIX, cutmix_minmax=config.AUG_CUTMIX_MINMAX,
            prob=config.AUG_MIXUP_PROB, switch_prob=config.AUG_MIXUP_SWITCH_PROB, mode=config.AUG_MIXUP_MODE,
            label_smoothing=config.RAF_LABEL_SMOOTHING, num_classes=config.RAF_NUM_CLASSES)
    return mixup_fn


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())
        
        self.epoch = 0

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    
def get_analysis_train_dataloader(data_choose, config) -> Iterable:

    transform = create_transform(
        input_size=config.img_size,
        scale=config.AUG_SCALE_SCALE if config.AUG_SCALE_SET else None,
        ratio=config.AUG_SCALE_RATIO if config.AUG_SCALE_SET else None,
        is_training=True,
        color_jitter=config.AUG_COLOR_JITTER if config.AUG_COLOR_JITTER > 0 else None,
        auto_augment=config.AUG_AUTO_AUGMENT if config.AUG_AUTO_AUGMENT != 'none' else None,
        re_prob=config.AUG_REPROB,
        re_mode=config.AUG_REMODE,
        re_count=config.AUG_RECOUNT,
        interpolation=config.INTERPOLATION,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5])

    if data_choose == "Expression":
        batch_size = config.batch_size
        if config.expression_train_dataset == "RAF-DB":
            dataset_train = RAFDBDataset(choose="train",
                                         data_path=config.RAF_data,
                                         label_path=config.RAF_label,
                                         train_sample_num=config.standard_train_sample_num,
                                         transform=transform,
                                         img_size=config.img_size)
        elif config.expression_train_dataset == "AffectNet":
            dataset_train = AffectNetDataset(choose="train",
                                         data_path=config.AffectNet_train_data,
                                         label_path=config.AffectNet_train_label,
                                         train_sample_num=config.standard_train_sample_num,
                                         transform=transform,
                                         img_size=config.img_size)
        elif config.expression_train_dataset == "FERPlus":
            dataset_train = FERPlusDataset(choose="train",
                                         data_path=config.FERPlus_train_data,
                                         label_path=config.FERPlus_train_label,
                                         train_sample_num=config.standard_train_sample_num,
                                         transform=transform,
                                         img_size=config.img_size)

    rank, world_size = get_dist_info()
    if config.expression_train_dataset == "AffectNet":
        sampler_train = ImbalancedDatasetSampler(dataset_train)
    else:
        sampler_train = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=config.train_num_workers,
        pin_memory=config.train_pin_memory,
        drop_last=True,
    )
    return data_loader_train


def get_analysis_val_dataloader(data_choose, config):

    if data_choose == "Expression":
        if config.expression_val_dataset == "RAF-DB":
            dataset_val = RAFDBDataset(choose="test",
                                         data_path=config.RAF_data,
                                         label_path=config.RAF_label,
                                         img_size=config.img_size)
        elif config.expression_val_dataset == "AffectNet":
            dataset_val = AffectNetDataset(choose="test",
                                         data_path=config.AffectNet_test_data,
                                         label_path=config.AffectNet_test_label,
                                         img_size=config.img_size)
        elif config.expression_val_dataset == "FERPlus":
            dataset_val = FERPlusDataset(choose="test",
                                         data_path=config.FERPlus_test_data,
                                         label_path=config.FERPlus_test_label,
                                         img_size=config.img_size)

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.val_num_workers,
        pin_memory=config.val_pin_memory,
        drop_last=False
    )

    return data_loader_val
