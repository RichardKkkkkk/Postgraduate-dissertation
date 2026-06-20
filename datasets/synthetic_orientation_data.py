import torch
from torch.utils.data import DataLoader, Dataset, Subset


SYNTHETIC_MEAN = (0.5, 0.5, 0.5)
SYNTHETIC_STD = (0.5, 0.5, 0.5)
ROW_CODE_TEMPLATES = {
    0: ((0, 2, 4), (1, 3, 5)),
    1: ((2, 5, 7), (0, 3, 6)),
}
COL_CODE_TEMPLATES = {
    0: ((0, 2, 4), (1, 3, 5)),
    1: ((2, 5, 7), (0, 3, 6)),
}


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


class SyntheticOrientationCleanDataset(Dataset):
    classes = ["horizontal", "vertical"]

    def __init__(
        self,
        split,
        dataset_size,
        image_size=32,
        line_width=3,
        noise_std=0.02,
        max_stripes=5,
        seed=42,
    ):
        self.split = split
        self.dataset_size = dataset_size
        self.image_size = image_size
        self.line_width = line_width
        self.noise_std = noise_std
        self.max_stripes = max_stripes
        self.seed = seed
        self.split_offset = {"train": 300_000, "val": 400_000, "test": 500_000}[split]

    def __len__(self):
        return self.dataset_size

    def _build_positions(self, generator):
        min_width = max(2, self.line_width - 1)
        max_width = max(min_width, self.line_width + 1)
        stripe_width = int(torch.randint(min_width, max_width + 1, (1,), generator=generator).item())
        num_stripes = int(torch.randint(2, self.max_stripes + 1, (1,), generator=generator).item())
        margin = max(2, stripe_width)
        usable = max(1, self.image_size - 2 * margin - stripe_width)
        candidates = torch.randperm(usable, generator=generator).tolist()

        positions = []
        min_gap = stripe_width + 2
        for candidate in candidates:
            start = margin + candidate
            if all(abs(start - existing) >= min_gap for existing in positions):
                positions.append(start)
            if len(positions) == num_stripes:
                break

        if not positions:
            positions = [margin]
        return positions, stripe_width

    def __getitem__(self, index):
        label = index % len(self.classes)
        generator = torch.Generator().manual_seed(self.seed + self.split_offset + index)

        background_intensity = torch.rand((1,), generator=generator).item() * 0.05
        stripe_intensity = 0.9 + torch.rand((1,), generator=generator).item() * 0.1
        image = torch.full((3, self.image_size, self.image_size), fill_value=background_intensity, dtype=torch.float32)

        positions, stripe_width = self._build_positions(generator)
        if label == 0:
            left_margin = int(torch.randint(0, max(1, self.image_size // 8 + 1), (1,), generator=generator).item())
            right_margin = int(torch.randint(0, max(1, self.image_size // 8 + 1), (1,), generator=generator).item())
            start_col = left_margin
            end_col = max(start_col + 1, self.image_size - right_margin)
            for start in positions:
                end = min(self.image_size, start + stripe_width)
                image[:, start:end, start_col:end_col] = stripe_intensity
        else:
            top_margin = int(torch.randint(0, max(1, self.image_size // 8 + 1), (1,), generator=generator).item())
            bottom_margin = int(torch.randint(0, max(1, self.image_size // 8 + 1), (1,), generator=generator).item())
            start_row = top_margin
            end_row = max(start_row + 1, self.image_size - bottom_margin)
            for start in positions:
                end = min(self.image_size, start + stripe_width)
                image[:, start_row:end_row, start:end] = stripe_intensity

        noise = torch.randn((3, self.image_size, self.image_size), generator=generator, dtype=torch.float32)
        image = (image + noise * self.noise_std).clamp(0.0, 1.0)
        image = (image - torch.tensor(SYNTHETIC_MEAN).view(3, 1, 1)) / torch.tensor(SYNTHETIC_STD).view(3, 1, 1)
        return image, label


class SyntheticOrientationHardDataset(Dataset):
    classes = ["horizontal", "vertical"]

    def __init__(
        self,
        split,
        dataset_size,
        image_size=32,
        line_width=3,
        noise_std=0.05,
        max_stripes=5,
        seed=42,
    ):
        self.split = split
        self.dataset_size = dataset_size
        self.image_size = image_size
        self.line_width = line_width
        self.noise_std = noise_std
        self.max_stripes = max_stripes
        self.seed = seed
        self.split_offset = {"train": 600_000, "val": 700_000, "test": 800_000}[split]

    def __len__(self):
        return self.dataset_size

    def _sample_stripe_layout(self, generator):
        min_width = max(2, self.line_width - 1)
        max_width = max(min_width, self.line_width + 2)
        stripe_width = int(torch.randint(min_width, max_width + 1, (1,), generator=generator).item())
        num_main = int(torch.randint(2, self.max_stripes + 1, (1,), generator=generator).item())
        margin = max(2, stripe_width)
        usable = max(1, self.image_size - 2 * margin - stripe_width)
        candidates = torch.randperm(usable, generator=generator).tolist()

        positions = []
        min_gap = stripe_width + 1
        for candidate in candidates:
            start = margin + candidate
            if all(abs(start - existing) >= min_gap for existing in positions):
                positions.append(start)
            if len(positions) == num_main:
                break

        if not positions:
            positions = [margin]
        return positions, stripe_width

    def _sample_span(self, generator, stripe_width, orientation):
        min_span = max(self.image_size // 2, stripe_width + 4)
        max_span = self.image_size - 2
        span = int(torch.randint(min_span, max_span + 1, (1,), generator=generator).item())
        start_limit = max(1, self.image_size - span)
        span_start = int(torch.randint(0, start_limit + 1, (1,), generator=generator).item())
        span_end = min(self.image_size, span_start + span)
        if orientation == "horizontal":
            return span_start, span_end, 0, self.image_size
        return 0, self.image_size, span_start, span_end

    def _paint_main_stripes(self, image, positions, stripe_width, label, generator):
        orientation = "horizontal" if label == 0 else "vertical"
        row_start, row_end, col_start, col_end = self._sample_span(generator, stripe_width, orientation)

        for start in positions:
            end = min(self.image_size, start + stripe_width)
            if label == 0:
                image[:, start:end, col_start:col_end] = 0.95
            else:
                image[:, row_start:row_end, start:end] = 0.95

    def _paint_distractors(self, image, stripe_width, label, generator):
        distractor_count = int(torch.randint(1, 3, (1,), generator=generator).item())
        distractor_intensity = 0.45 + torch.rand((1,), generator=generator).item() * 0.18

        for _ in range(distractor_count):
            short_span = int(
                torch.randint(
                    max(stripe_width + 2, self.image_size // 4),
                    max(stripe_width + 3, self.image_size // 2) + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            short_start = int(torch.randint(0, max(1, self.image_size - short_span + 1), (1,), generator=generator).item())
            short_end = min(self.image_size, short_start + short_span)
            start = int(torch.randint(0, max(1, self.image_size - stripe_width + 1), (1,), generator=generator).item())
            end = min(self.image_size, start + stripe_width)

            if label == 0:
                image[:, short_start:short_end, start:end] = distractor_intensity
            else:
                image[:, start:end, short_start:short_end] = distractor_intensity

    def _paint_occluder(self, image, generator):
        if torch.rand((1,), generator=generator).item() < 0.7:
            occ_height = int(torch.randint(self.image_size // 6, self.image_size // 3 + 1, (1,), generator=generator).item())
            occ_width = int(torch.randint(self.image_size // 6, self.image_size // 3 + 1, (1,), generator=generator).item())
            top = int(torch.randint(0, max(1, self.image_size - occ_height + 1), (1,), generator=generator).item())
            left = int(torch.randint(0, max(1, self.image_size - occ_width + 1), (1,), generator=generator).item())
            image[:, top : top + occ_height, left : left + occ_width] *= 0.15

    def __getitem__(self, index):
        label = index % len(self.classes)
        generator = torch.Generator().manual_seed(self.seed + self.split_offset + index)

        background_intensity = 0.03 + torch.rand((1,), generator=generator).item() * 0.06
        image = torch.full((3, self.image_size, self.image_size), fill_value=background_intensity, dtype=torch.float32)

        positions, stripe_width = self._sample_stripe_layout(generator)
        self._paint_main_stripes(image, positions, stripe_width, label, generator)
        self._paint_distractors(image, stripe_width, label, generator)
        self._paint_occluder(image, generator)

        noise = torch.randn((3, self.image_size, self.image_size), generator=generator, dtype=torch.float32)
        image = (image + noise * self.noise_std).clamp(0.0, 1.0)
        image = (image - torch.tensor(SYNTHETIC_MEAN).view(3, 1, 1)) / torch.tensor(SYNTHETIC_STD).view(3, 1, 1)
        return image, label


class SyntheticAxisCodeDataset(Dataset):
    classes = ["template_a", "template_b"]

    def __init__(
        self,
        split,
        dataset_size,
        target_axis,
        image_size=32,
        noise_std=0.03,
        seed=42,
    ):
        self.split = split
        self.dataset_size = dataset_size
        self.target_axis = target_axis
        self.image_size = image_size
        self.noise_std = noise_std
        self.seed = seed
        self.cell_size = 4
        if image_size % self.cell_size != 0:
            raise ValueError("SyntheticAxisCodeDataset requires image_size to be divisible by 4.")
        self.grid_size = image_size // self.cell_size
        if self.grid_size < 8:
            raise ValueError("SyntheticAxisCodeDataset expects at least an 8x8 patch grid.")
        self.split_offset = {
            "train": 900_000,
            "val": 1_000_000,
            "test": 1_100_000,
        }[split]

    def __len__(self):
        return self.dataset_size

    def _paint_cell(self, image, row, col, intensity):
        row_start = row * self.cell_size
        row_end = row_start + self.cell_size
        col_start = col * self.cell_size
        col_end = col_start + self.cell_size
        image[:, row_start:row_end, col_start:col_end] = intensity

    def _sample_template(self, label, generator):
        template_map = ROW_CODE_TEMPLATES if self.target_axis == "row" else COL_CODE_TEMPLATES
        template_bank = template_map[label]
        template_index = int(torch.randint(0, len(template_bank), (1,), generator=generator).item())
        return list(template_bank[template_index])

    def _paint_signal(self, image, label, generator):
        signal_intensity = 0.9 + torch.rand((1,), generator=generator).item() * 0.1
        template_positions = self._sample_template(label, generator)

        if self.target_axis == "row":
            for row in template_positions:
                for col in range(self.grid_size):
                    self._paint_cell(image, row=row, col=col, intensity=signal_intensity)
        else:
            for col in template_positions:
                for row in range(self.grid_size):
                    self._paint_cell(image, row=row, col=col, intensity=signal_intensity)

    def __getitem__(self, index):
        label = index % len(self.classes)
        generator = torch.Generator().manual_seed(self.seed + self.split_offset + index)

        background_intensity = torch.rand((1,), generator=generator).item() * 0.04
        image = torch.full((3, self.image_size, self.image_size), fill_value=background_intensity, dtype=torch.float32)

        self._paint_signal(image, label, generator)

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


def build_synthetic_orientation_clean_dataloaders(
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
    noise_std=0.02,
    max_stripes=5,
):
    train_dataset = SyntheticOrientationCleanDataset(
        split="train",
        dataset_size=train_size,
        image_size=image_size,
        line_width=line_width,
        noise_std=noise_std,
        max_stripes=max_stripes,
        seed=seed,
    )
    val_dataset = SyntheticOrientationCleanDataset(
        split="val",
        dataset_size=val_size,
        image_size=image_size,
        line_width=line_width,
        noise_std=noise_std,
        max_stripes=max_stripes,
        seed=seed,
    )
    test_dataset = SyntheticOrientationCleanDataset(
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


def build_synthetic_orientation_hard_dataloaders(
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
    noise_std=0.05,
    max_stripes=5,
):
    train_dataset = SyntheticOrientationHardDataset(
        split="train",
        dataset_size=train_size,
        image_size=image_size,
        line_width=line_width,
        noise_std=noise_std,
        max_stripes=max_stripes,
        seed=seed,
    )
    val_dataset = SyntheticOrientationHardDataset(
        split="val",
        dataset_size=val_size,
        image_size=image_size,
        line_width=line_width,
        noise_std=noise_std,
        max_stripes=max_stripes,
        seed=seed,
    )
    test_dataset = SyntheticOrientationHardDataset(
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


def build_synthetic_axis_code_dataloaders(
    batch_size,
    train_subset,
    val_subset,
    test_subset,
    num_workers,
    seed,
    train_size,
    val_size,
    test_size,
    target_axis,
    image_size=32,
    noise_std=0.03,
):
    train_dataset = SyntheticAxisCodeDataset(
        split="train",
        dataset_size=train_size,
        target_axis=target_axis,
        image_size=image_size,
        noise_std=noise_std,
        seed=seed,
    )
    val_dataset = SyntheticAxisCodeDataset(
        split="val",
        dataset_size=val_size,
        target_axis=target_axis,
        image_size=image_size,
        noise_std=noise_std,
        seed=seed,
    )
    test_dataset = SyntheticAxisCodeDataset(
        split="test",
        dataset_size=test_size,
        target_axis=target_axis,
        image_size=image_size,
        noise_std=noise_std,
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


def build_synthetic_row_code_dataloaders(
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
    noise_std=0.03,
):
    return build_synthetic_axis_code_dataloaders(
        batch_size=batch_size,
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
        num_workers=num_workers,
        seed=seed,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        target_axis="row",
        image_size=image_size,
        noise_std=noise_std,
    )


def build_synthetic_col_code_dataloaders(
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
    noise_std=0.03,
):
    return build_synthetic_axis_code_dataloaders(
        batch_size=batch_size,
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
        num_workers=num_workers,
        seed=seed,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        target_axis="col",
        image_size=image_size,
        noise_std=noise_std,
    )
