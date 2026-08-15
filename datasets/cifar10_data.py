import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_subset(dataset, subset_size, seed):
    if subset_size is None or subset_size >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def split_train_val_indices(dataset_size, val_ratio, seed):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    val_size = max(1, int(dataset_size * val_ratio))
    if val_size >= dataset_size:
        raise ValueError("val_ratio leaves no training samples.")

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def unwrap_dataset(dataset):
    current = dataset
    while isinstance(current, Subset):
        current = current.dataset
    return current


def get_class_names(dataset):
    base_dataset = unwrap_dataset(dataset)
    if hasattr(base_dataset, "classes"):
        return list(base_dataset.classes)
    return [str(index) for index in range(len(base_dataset))]


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
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    return build_cifar10_split_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
        num_workers=num_workers,
        seed=seed,
        split_seed=split_seed,
        val_ratio=val_ratio,
        train_transform=transform_train,
        eval_transform=transform_test,
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
    mean = IMAGENET_MEAN if use_imagenet_norm else CIFAR10_MEAN
    std = IMAGENET_STD if use_imagenet_norm else CIFAR10_STD

    transform_train = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return build_cifar10_split_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
        num_workers=num_workers,
        seed=seed,
        split_seed=split_seed,
        val_ratio=val_ratio,
        train_transform=transform_train,
        eval_transform=transform_test,
    )


def build_cifar10_split_loaders(
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
    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    full_val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=eval_transform,
    )
    test_dataset = datasets.CIFAR10(
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
