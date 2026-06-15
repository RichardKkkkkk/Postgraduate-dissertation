import torch
from torch.utils.data import DataLoader, Dataset, Subset


SYNTHETIC_MEAN = (0.5, 0.5, 0.5)
SYNTHETIC_STD = (0.5, 0.5, 0.5)


def make_subset(dataset, subset_size, seed):
    if subset_size is None or subset_size >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


class SyntheticOrientationDataset(Dataset):
    classes = ["horizontal", "vertical"]

    def __init__(
        self,
        split,
        dataset_size,
        image_size=32,
        line_width=3,
        noise_std=0.08,
        max_stripes=4,
        seed=42,
    ):
        self.split = split
        self.dataset_size = dataset_size
        self.image_size = image_size
        self.line_width = line_width
        self.noise_std = noise_std
        self.max_stripes = max_stripes
        self.seed = seed
        self.split_offset = {"train": 0, "val": 100_000, "test": 200_000}[split]

    def __len__(self):
        return self.dataset_size

    def _build_positions(self, generator):
        num_stripes = int(torch.randint(1, self.max_stripes + 1, (1,), generator=generator).item())
        min_gap = max(2, self.line_width)
        candidates = torch.randperm(self.image_size - self.line_width + 1, generator=generator).tolist()

        positions = []
        for candidate in candidates:
            if all(abs(candidate - existing) >= min_gap for existing in positions):
                positions.append(candidate)
            if len(positions) == num_stripes:
                break

        if not positions:
            positions = [0]
        return positions

    def __getitem__(self, index):
        label = index % len(self.classes)
        generator = torch.Generator().manual_seed(self.seed + self.split_offset + index)

        base_intensity = torch.rand((1,), generator=generator).item() * 0.1
        stripe_intensity = 0.75 + torch.rand((1,), generator=generator).item() * 0.25
        image = torch.full((3, self.image_size, self.image_size), fill_value=base_intensity, dtype=torch.float32)

        positions = self._build_positions(generator)
        for start in positions:
            end = min(self.image_size, start + self.line_width)
            if label == 0:
                image[:, start:end, :] = stripe_intensity
            else:
                image[:, :, start:end] = stripe_intensity

        noise = torch.randn((3, self.image_size, self.image_size), generator=generator, dtype=torch.float32)
        image = (image + noise * self.noise_std).clamp(0.0, 1.0)
        image = (image - torch.tensor(SYNTHETIC_MEAN).view(3, 1, 1)) / torch.tensor(SYNTHETIC_STD).view(3, 1, 1)

        return image, label


def build_synthetic_orientation_dataloaders(
    batch_size,
    train_subset,
    val_subset,
    test_subset,
    num_workers,
    seed,
    train_size,
    val_size,
    test_size,
    image_size=32,
    line_width=3,
    noise_std=0.08,
    max_stripes=4,
):
    train_dataset = SyntheticOrientationDataset(
        split="train",
        dataset_size=train_size,
        image_size=image_size,
        line_width=line_width,
        noise_std=noise_std,
        max_stripes=max_stripes,
        seed=seed,
    )
    val_dataset = SyntheticOrientationDataset(
        split="val",
        dataset_size=val_size,
        image_size=image_size,
        line_width=line_width,
        noise_std=noise_std,
        max_stripes=max_stripes,
        seed=seed,
    )
    test_dataset = SyntheticOrientationDataset(
        split="test",
        dataset_size=test_size,
        image_size=image_size,
        line_width=line_width,
        noise_std=noise_std,
        max_stripes=max_stripes,
        seed=seed,
    )

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
