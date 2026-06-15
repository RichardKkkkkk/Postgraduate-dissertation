import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TARGET_CLASSES = ("horizontal", "vertical")
ELEMENT_ANNOTATION_FILENAME = "composition_elements.json"
SPLIT_FILENAME = "split.json"
CLASS_CONTAINER_KEYS = {
    "composition_class",
    "composition_classes",
    "class",
    "classes",
    "composition_attribute",
    "composition_attributes",
    "attributes",
}


def make_subset(dataset, subset_size, seed):
    if subset_size is None or subset_size >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def _normalize_token(value):
    return str(value).strip().lower().replace("-", " ").replace("_", " ")


def _is_positive_value(value, target_name=None):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        normalized = _normalize_token(value)
        allowed = {"1", "true", "yes", "present", "positive"}
        if target_name is not None:
            allowed.add(_normalize_token(target_name))
        return normalized in allowed
    if isinstance(value, (list, tuple, set)):
        return any(_is_positive_value(item, target_name=target_name) for item in value)
    if isinstance(value, dict):
        return any(_is_positive_value(item, target_name=target_name) for item in value.values())
    return False


def _extract_target_classes(record):
    found = set()

    def visit(node, parent_key=None):
        if isinstance(node, dict):
            for key, value in node.items():
                normalized_key = _normalize_token(key)

                if normalized_key in TARGET_CLASSES and _is_positive_value(value, target_name=normalized_key):
                    found.add(normalized_key)

                if normalized_key in CLASS_CONTAINER_KEYS:
                    visit_class_container(value)
                else:
                    visit(value, parent_key=normalized_key)
            return

        if isinstance(node, (list, tuple, set)):
            for item in node:
                visit(item, parent_key=parent_key)
            return

        if isinstance(node, str):
            normalized = _normalize_token(node)
            if normalized in TARGET_CLASSES:
                found.add(normalized)

    def visit_class_container(container):
        if isinstance(container, str):
            normalized = _normalize_token(container)
            if normalized in TARGET_CLASSES:
                found.add(normalized)
            return

        if isinstance(container, (list, tuple, set)):
            for item in container:
                visit_class_container(item)
            return

        if isinstance(container, dict):
            for key, value in container.items():
                normalized_key = _normalize_token(key)
                if normalized_key in TARGET_CLASSES and _is_positive_value(value, target_name=normalized_key):
                    found.add(normalized_key)
                elif isinstance(value, str):
                    normalized_value = _normalize_token(value)
                    if normalized_value in TARGET_CLASSES:
                        found.add(normalized_value)
                else:
                    visit_class_container(value)
            return

        visit(container)

    visit(record)
    return found


def _resolve_image_name(image_id, record):
    candidates = []
    if isinstance(record, dict):
        for key in ("image_name", "image_path", "filename", "file_name", "img_name", "img", "id", "image_id"):
            if key in record:
                candidates.append(record[key])
    if image_id is not None:
        candidates.append(image_id)

    for candidate in candidates:
        if candidate is None:
            continue
        name = str(candidate).strip()
        if not name:
            continue
        path = Path(name)
        if path.suffix:
            return path.name
        return f"{path.name}.jpg"

    raise ValueError(f"Unable to resolve image name from record: {record}")


def _iter_annotation_records(annotation_data):
    if isinstance(annotation_data, dict):
        for image_id, record in annotation_data.items():
            if isinstance(record, dict):
                yield image_id, record
            else:
                yield image_id, {"composition_classes": record}
        return

    if isinstance(annotation_data, list):
        for index, record in enumerate(annotation_data):
            if not isinstance(record, dict):
                record = {"composition_classes": record}
            yield str(index), record
        return

    raise ValueError("Unsupported CADB annotation format.")


def load_cadb_orientation_samples(cadb_root, label_mode="exclusive"):
    cadb_root = Path(cadb_root)
    images_dir = cadb_root / "images"
    element_annotation_path = cadb_root / ELEMENT_ANNOTATION_FILENAME
    split_path = cadb_root / SPLIT_FILENAME
    annotation_candidates = [
        cadb_root / "composition_classes.json",
        cadb_root / "composition_attributes.json",
    ]
    annotation_path = next((path for path in annotation_candidates if path.exists()), None)
    if annotation_path is None and not element_annotation_path.exists():
        raise FileNotFoundError(
            "Could not find CADB annotation file. Expected one of: "
            f"{', '.join(str(path.name) for path in [*annotation_candidates, element_annotation_path])}"
        )
    if not images_dir.exists():
        raise FileNotFoundError(f"Could not find CADB images directory: {images_dir}")

    if element_annotation_path.exists():
        with element_annotation_path.open("r", encoding="utf-8") as handle:
            element_annotation = json.load(handle)
        split_data = None
        if split_path.exists():
            with split_path.open("r", encoding="utf-8") as handle:
                split_data = json.load(handle)

        samples = []
        for image_name, element_record in element_annotation.items():
            if not isinstance(element_record, dict):
                continue

            has_horizontal = "horizontal" in element_record and len(element_record["horizontal"]) > 0
            has_vertical = "vertical" in element_record and len(element_record["vertical"]) > 0

            if label_mode == "exclusive":
                if has_horizontal == has_vertical:
                    continue
            elif label_mode == "inclusive":
                if not (has_horizontal or has_vertical):
                    continue
                if has_horizontal and has_vertical:
                    continue
            else:
                raise ValueError(f"Unsupported CADB label mode: {label_mode}")

            label = 0 if has_horizontal else 1
            image_path = images_dir / image_name
            if not image_path.exists():
                continue

            split_name = None
            if split_data is not None:
                if image_name in split_data.get("train", []):
                    split_name = "train"
                elif image_name in split_data.get("test", []):
                    split_name = "test"

            samples.append((image_path, label, split_name))

        if not samples:
            raise ValueError(
                "No CADB orientation samples were found from composition_elements.json. "
                "Check the label mode and annotation files."
            )

        return samples

    with annotation_path.open("r", encoding="utf-8") as handle:
        annotation_data = json.load(handle)

    samples = []
    for image_id, record in _iter_annotation_records(annotation_data):
        target_classes = _extract_target_classes(record)
        has_horizontal = "horizontal" in target_classes
        has_vertical = "vertical" in target_classes

        if label_mode == "exclusive":
            if has_horizontal == has_vertical:
                continue
        elif label_mode == "inclusive":
            if not (has_horizontal or has_vertical):
                continue
        else:
            raise ValueError(f"Unsupported CADB label mode: {label_mode}")

        if has_horizontal and not has_vertical:
            label = 0
        elif has_vertical and not has_horizontal:
            label = 1
        else:
            continue

        image_name = _resolve_image_name(image_id, record)
        image_path = images_dir / image_name
        if not image_path.exists():
            continue

        samples.append((image_path, label, None))

    if not samples:
        raise ValueError(
            "No CADB orientation samples were found. "
            "Check the annotation file format and the label mode."
        )

    return samples


def stratified_split_indices(labels, val_ratio, test_ratio, seed):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1.")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.")

    label_to_indices = {}
    for index, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(index)

    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    for label_indices in label_to_indices.values():
        shuffled = torch.tensor(label_indices)[
            torch.randperm(len(label_indices), generator=generator)
        ].tolist()
        num_items = len(shuffled)
        test_count = max(1, int(num_items * test_ratio))
        remaining = num_items - test_count
        if remaining < 2:
            raise ValueError("Not enough CADB samples per class after test split.")

        val_count = max(1, int(remaining * val_ratio))
        train_count = remaining - val_count
        if train_count < 1:
            raise ValueError("Not enough CADB samples per class after validation split.")

        test_indices.extend(shuffled[:test_count])
        val_indices.extend(shuffled[test_count : test_count + val_count])
        train_indices.extend(shuffled[test_count + val_count :])

    train_indices.sort()
    val_indices.sort()
    test_indices.sort()
    return train_indices, val_indices, test_indices


def split_train_val_from_indices(labels, train_pool_indices, val_ratio, seed):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    label_to_indices = {}
    for index in train_pool_indices:
        label = labels[index]
        label_to_indices.setdefault(label, []).append(index)

    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    val_indices = []

    for label_indices in label_to_indices.values():
        shuffled = torch.tensor(label_indices)[
            torch.randperm(len(label_indices), generator=generator)
        ].tolist()
        val_count = max(1, int(len(shuffled) * val_ratio))
        if val_count >= len(shuffled):
            val_count = len(shuffled) - 1
        if val_count < 1:
            raise ValueError("Not enough CADB train samples per class to form a validation split.")

        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices


class CADBOrientationDataset(Dataset):
    classes = ["horizontal", "vertical"]

    def __init__(self, samples, transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label


def build_cadb_orientation_dataloaders(
    cadb_root,
    batch_size,
    train_subset,
    val_subset,
    test_subset,
    num_workers,
    seed,
    val_ratio,
    test_ratio,
    image_size=96,
    label_mode="exclusive",
    use_imagenet_norm=True,
):
    mean = IMAGENET_MEAN if use_imagenet_norm else (0.5, 0.5, 0.5)
    std = IMAGENET_STD if use_imagenet_norm else (0.5, 0.5, 0.5)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    samples_with_split = load_cadb_orientation_samples(cadb_root=cadb_root, label_mode=label_mode)
    labels = [label for _, label, _ in samples_with_split]

    explicit_train_indices = [index for index, (_, _, split_name) in enumerate(samples_with_split) if split_name == "train"]
    explicit_test_indices = [index for index, (_, _, split_name) in enumerate(samples_with_split) if split_name == "test"]

    if explicit_train_indices and explicit_test_indices:
        train_indices, val_indices = split_train_val_from_indices(
            labels=labels,
            train_pool_indices=explicit_train_indices,
            val_ratio=val_ratio,
            seed=seed,
        )
        test_indices = sorted(explicit_test_indices)
    else:
        train_indices, val_indices, test_indices = stratified_split_indices(
            labels=labels,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    samples = [(image_path, label) for image_path, label, _ in samples_with_split]
    full_dataset = CADBOrientationDataset(samples=samples, transform=transform)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

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
