import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .cifar10_data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    make_subset,
    split_train_val_indices,
)


# Channel statistics computed from the CIFAR-100 training images and commonly
# used for CIFAR-100 classification.  They are fixed before any model training.
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def build_vit_dataloaders(
    data_dir,
    batch_size,
    train_subset,
    val_subset,
    test_subset,
    num_workers,
    seed,
    val_ratio,
    split_seed=None,
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    return build_cifar100_split_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
        num_workers=num_workers,
        seed=seed,
        split_seed=split_seed,
        val_ratio=val_ratio,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


def build_resnet_dataloaders(
    data_dir,
    batch_size,
    train_subset,
    val_subset,
    test_subset,
    num_workers,
    seed,
    val_ratio,
    split_seed,
    image_size,
    use_imagenet_norm,
):
    mean = IMAGENET_MEAN if use_imagenet_norm else CIFAR100_MEAN
    std = IMAGENET_STD if use_imagenet_norm else CIFAR100_STD
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return build_cifar100_split_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
        num_workers=num_workers,
        seed=seed,
        split_seed=split_seed,
        val_ratio=val_ratio,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


def build_cifar100_split_loaders(
    data_dir,
    batch_size,
    train_subset,
    val_subset,
    test_subset,
    num_workers,
    seed,
    split_seed,
    val_ratio,
    train_transform,
    eval_transform,
):
    full_train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    full_val_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=False,
        transform=eval_transform,
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    effective_split_seed = seed if split_seed is None else split_seed
    train_indices, val_indices = split_train_val_indices(
        dataset_size=len(full_train_dataset),
        val_ratio=val_ratio,
        seed=effective_split_seed,
    )
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    train_dataset = make_subset(train_dataset, train_subset, seed)
    val_dataset = make_subset(val_dataset, val_subset, seed)
    test_dataset = make_subset(test_dataset, test_subset, seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader
