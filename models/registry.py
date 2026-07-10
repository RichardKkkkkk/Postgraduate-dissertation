import argparse
from dataclasses import dataclass, field
from typing import Any, Callable

import torch.nn as nn
from torchvision import models

from datasets.cadb_data import (
    CADB_ELEMENT_LABELS,
    build_cadb_elements_multilabel_dataloaders,
    build_cadb_orientation_dataloaders,
    build_cadb_scene_dataloaders,
)
from datasets.cifar10_data import build_resnet_dataloaders, build_vit_dataloaders
from datasets.synthetic_orientation_data import (
    build_synthetic_col_code_dataloaders,
    build_synthetic_orientation_clean_dataloaders,
    build_synthetic_orientation_hard_dataloaders,
    build_synthetic_orientation_dataloaders,
    build_synthetic_row_code_dataloaders,
)
from .vit import ViT
from .vit_axis_sinusoidal import (
    ViTAdditiveSinusoidal,
    ViTAdditiveSinusoidalShifted,
    ViTColSinusoidal,
    ViTMultiplicativeSinusoidal,
    ViTMultiplicativeSinusoidalShifted,
    ViTRadialSinusoidal,
    ViTRowSinusoidal,
    ViTSquaredMultiplicativeSinusoidal,
    ViTSquaredMultiplicativeSinusoidalShifted,
)
from .vit_baseline import ViTBaseline
from .vit_rope import ViTRoPE
from .vit_rope_2d import ViTRoPE2D


@dataclass(frozen=True)
class ExperimentSpec:
    model_name: str
    architecture: str
    variant: str
    plot_title_prefix: str
    defaults: dict[str, Any]
    build_model: Callable[[argparse.Namespace], tuple[nn.Module, dict[str, Any]]]
    build_dataloaders: Callable[[argparse.Namespace], tuple[Any, Any, Any]]
    extra_summary_fields: dict[str, Any] = field(default_factory=dict)


EXPERIMENT_REGISTRY: dict[str, ExperimentSpec] = {}
SUPPORTED_DATASETS = (
    "cifar10",
    "synthetic_orientation",
    "synthetic_orientation_clean",
    "synthetic_orientation_hard",
    "synthetic_row_code",
    "synthetic_col_code",
    "cadb_orientation",
    "cadb_scene",
    "cadb_elements",
)
DATASET_NUM_CLASSES = {
    "cifar10": 10,
    "synthetic_orientation": 2,
    "synthetic_orientation_clean": 2,
    "synthetic_orientation_hard": 2,
    "synthetic_row_code": 2,
    "synthetic_col_code": 2,
    "cadb_orientation": 2,
    "cadb_scene": 10,
    "cadb_elements": len(CADB_ELEMENT_LABELS),
}
DATASET_DEFAULT_IMAGE_SIZE = {
    "cifar10": 32,
    "synthetic_orientation": 32,
    "synthetic_orientation_clean": 32,
    "synthetic_orientation_hard": 32,
    "synthetic_row_code": 32,
    "synthetic_col_code": 32,
    "cadb_orientation": 96,
    "cadb_scene": 96,
    "cadb_elements": 96,
}
DATASET_DISPLAY_NAMES = {
    "cifar10": "CIFAR-10",
    "synthetic_orientation": "Synthetic Orientation",
    "synthetic_orientation_clean": "Synthetic Orientation Clean",
    "synthetic_orientation_hard": "Synthetic Orientation Hard",
    "synthetic_row_code": "Synthetic Row-Code",
    "synthetic_col_code": "Synthetic Column-Code",
    "cadb_orientation": "CADB Orientation",
    "cadb_scene": "CADB Scene Categories",
    "cadb_elements": "CADB Composition Elements",
}


def register_experiment(spec: ExperimentSpec):
    if spec.model_name in EXPERIMENT_REGISTRY:
        raise ValueError(f"Duplicate experiment model name: {spec.model_name}")
    EXPERIMENT_REGISTRY[spec.model_name] = spec


def get_dataset_num_classes(dataset_name):
    return DATASET_NUM_CLASSES[dataset_name]


def get_dataset_image_size(args):
    return args.image_size or DATASET_DEFAULT_IMAGE_SIZE[args.dataset]


def get_dataset_display_name(dataset_name):
    return DATASET_DISPLAY_NAMES[dataset_name]


def build_vit_model_config(args):
    return {
        "img_size": get_dataset_image_size(args),
        "patch_size": 4,
        "in_channels": 3,
        "num_classes": get_dataset_num_classes(args.dataset),
        "embed_dim": 128,
        "num_blocks": 4,
        "num_heads": 4,
        "mlp_hidden_dim": 512,
        "embedding_dropout": args.embedding_dropout,
        "attention_dropout": args.attention_dropout,
        "projection_dropout": args.projection_dropout,
        "mlp_dropout": args.mlp_dropout,
    }


def build_vit_learnable_position_model(args):
    model_config = build_vit_model_config(args)
    return ViT(**model_config), model_config


def build_vit_baseline_model(args):
    model_config = build_vit_model_config(args)
    return ViTBaseline(**model_config), model_config


def build_vit_rope_model(args):
    model_config = build_vit_model_config(args)
    model_config["rope_base"] = args.rope_base
    return ViTRoPE(**model_config), model_config


def build_vit_rope_2d_model(args):
    model_config = build_vit_model_config(args)
    model_config["rope_base"] = args.rope_base
    return ViTRoPE2D(**model_config), model_config


def build_vit_row_sinusoidal_model(args):
    model_config = build_vit_model_config(args)
    return ViTRowSinusoidal(**model_config), model_config


def build_vit_col_sinusoidal_model(args):
    model_config = build_vit_model_config(args)
    return ViTColSinusoidal(**model_config), model_config


def build_vit_radial_sinusoidal_model(args):
    model_config = build_vit_model_config(args)
    return ViTRadialSinusoidal(**model_config), model_config


def build_vit_additive_sinusoidal_model(args):
    model_config = build_vit_model_config(args)
    return ViTAdditiveSinusoidal(**model_config), model_config


def build_vit_additive_sinusoidal_shifted_model(args):
    model_config = build_vit_model_config(args)
    return ViTAdditiveSinusoidalShifted(**model_config), model_config


def build_vit_multiplicative_sinusoidal_model(args):
    model_config = build_vit_model_config(args)
    return ViTMultiplicativeSinusoidal(**model_config), model_config


def build_vit_multiplicative_sinusoidal_shifted_model(args):
    model_config = build_vit_model_config(args)
    return ViTMultiplicativeSinusoidalShifted(**model_config), model_config


def build_vit_squared_multiplicative_sinusoidal_model(args):
    model_config = build_vit_model_config(args)
    return ViTSquaredMultiplicativeSinusoidal(**model_config), model_config


def build_vit_squared_multiplicative_sinusoidal_shifted_model(args):
    model_config = build_vit_model_config(args)
    return ViTSquaredMultiplicativeSinusoidalShifted(**model_config), model_config


def build_vit_family_dataloaders(args):
    if args.dataset == "cifar10":
        return build_vit_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            val_ratio=args.val_ratio,
        )
    if args.dataset == "synthetic_orientation":
        return build_synthetic_orientation_dataloaders(
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            train_size=args.synthetic_train_size,
            val_size=args.synthetic_val_size,
            test_size=args.synthetic_test_size,
            image_size=get_dataset_image_size(args),
            line_width=args.synthetic_line_width,
            noise_std=args.synthetic_noise_std,
            max_stripes=args.synthetic_max_stripes,
        )
    if args.dataset == "synthetic_orientation_clean":
        return build_synthetic_orientation_clean_dataloaders(
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            train_size=args.synthetic_train_size,
            val_size=args.synthetic_val_size,
            test_size=args.synthetic_test_size,
            image_size=get_dataset_image_size(args),
            line_width=args.synthetic_line_width,
            noise_std=args.synthetic_noise_std,
            max_stripes=args.synthetic_max_stripes,
        )
    if args.dataset == "synthetic_orientation_hard":
        return build_synthetic_orientation_hard_dataloaders(
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            train_size=args.synthetic_train_size,
            val_size=args.synthetic_val_size,
            test_size=args.synthetic_test_size,
            image_size=get_dataset_image_size(args),
            line_width=args.synthetic_line_width,
            noise_std=args.synthetic_noise_std,
            max_stripes=args.synthetic_max_stripes,
        )
    if args.dataset == "synthetic_row_code":
        return build_synthetic_row_code_dataloaders(
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            train_size=args.synthetic_train_size,
            val_size=args.synthetic_val_size,
            test_size=args.synthetic_test_size,
            image_size=get_dataset_image_size(args),
            noise_std=args.synthetic_noise_std,
        )
    if args.dataset == "synthetic_col_code":
        return build_synthetic_col_code_dataloaders(
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            train_size=args.synthetic_train_size,
            val_size=args.synthetic_val_size,
            test_size=args.synthetic_test_size,
            image_size=get_dataset_image_size(args),
            noise_std=args.synthetic_noise_std,
        )
    if args.dataset == "cadb_orientation":
        return build_cadb_orientation_dataloaders(
            cadb_root=args.cadb_root or (args.data_dir / "CADB_Dataset"),
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.cadb_test_ratio,
            image_size=get_dataset_image_size(args),
            label_mode=args.cadb_label_mode,
            balance_mode=args.cadb_balance_mode,
            use_imagenet_norm=True,
        )
    if args.dataset == "cadb_scene":
        return build_cadb_scene_dataloaders(
            cadb_root=args.cadb_root or (args.data_dir / "CADB_Dataset"),
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.cadb_test_ratio,
            image_size=get_dataset_image_size(args),
            use_imagenet_norm=True,
        )
    if args.dataset == "cadb_elements":
        return build_cadb_elements_multilabel_dataloaders(
            cadb_root=args.cadb_root or (args.data_dir / "CADB_Dataset"),
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            val_ratio=args.val_ratio,
            image_size=get_dataset_image_size(args),
            use_imagenet_norm=True,
        )
    raise ValueError(f"Unsupported dataset for ViT family: {args.dataset}")


def build_resnet18_model(weights_name, num_classes=10):
    if weights_name == "imagenet":
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet18_scratch_model(args):
    image_size = get_dataset_image_size(args)
    num_classes = get_dataset_num_classes(args.dataset)
    model = build_resnet18_model("none", num_classes=num_classes)
    model_config = {
        "architecture": "resnet18",
        "variant": "scratch",
        "weights": "none",
        "image_size": image_size,
        "num_classes": num_classes,
    }
    return model, model_config


def build_resnet18_imagenet_model(args):
    image_size = args.image_size or 224
    num_classes = get_dataset_num_classes(args.dataset)
    model = build_resnet18_model("imagenet", num_classes=num_classes)
    model_config = {
        "architecture": "resnet18",
        "variant": "imagenet",
        "weights": "imagenet",
        "image_size": image_size,
        "num_classes": num_classes,
    }
    return model, model_config


def build_resnet18_scratch_dataloaders(args):
    image_size = get_dataset_image_size(args)
    if args.dataset == "cifar10":
        return build_resnet_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            val_ratio=args.val_ratio,
            image_size=image_size,
            use_imagenet_norm=False,
        )
    if args.dataset == "synthetic_orientation":
        return build_synthetic_orientation_dataloaders(
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            train_size=args.synthetic_train_size,
            val_size=args.synthetic_val_size,
            test_size=args.synthetic_test_size,
            image_size=image_size,
            line_width=args.synthetic_line_width,
            noise_std=args.synthetic_noise_std,
            max_stripes=args.synthetic_max_stripes,
        )
    if args.dataset == "synthetic_orientation_clean":
        return build_synthetic_orientation_clean_dataloaders(
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            train_size=args.synthetic_train_size,
            val_size=args.synthetic_val_size,
            test_size=args.synthetic_test_size,
            image_size=image_size,
            line_width=args.synthetic_line_width,
            noise_std=args.synthetic_noise_std,
            max_stripes=args.synthetic_max_stripes,
        )
    if args.dataset == "synthetic_orientation_hard":
        return build_synthetic_orientation_hard_dataloaders(
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            train_size=args.synthetic_train_size,
            val_size=args.synthetic_val_size,
            test_size=args.synthetic_test_size,
            image_size=image_size,
            line_width=args.synthetic_line_width,
            noise_std=args.synthetic_noise_std,
            max_stripes=args.synthetic_max_stripes,
        )
    if args.dataset == "synthetic_row_code":
        return build_synthetic_row_code_dataloaders(
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            train_size=args.synthetic_train_size,
            val_size=args.synthetic_val_size,
            test_size=args.synthetic_test_size,
            image_size=image_size,
            noise_std=args.synthetic_noise_std,
        )
    if args.dataset == "synthetic_col_code":
        return build_synthetic_col_code_dataloaders(
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            train_size=args.synthetic_train_size,
            val_size=args.synthetic_val_size,
            test_size=args.synthetic_test_size,
            image_size=image_size,
            noise_std=args.synthetic_noise_std,
        )
    if args.dataset == "cadb_orientation":
        return build_cadb_orientation_dataloaders(
            cadb_root=args.cadb_root or (args.data_dir / "CADB_Dataset"),
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.cadb_test_ratio,
            image_size=image_size,
            label_mode=args.cadb_label_mode,
            balance_mode=args.cadb_balance_mode,
            use_imagenet_norm=False,
        )
    if args.dataset == "cadb_scene":
        return build_cadb_scene_dataloaders(
            cadb_root=args.cadb_root or (args.data_dir / "CADB_Dataset"),
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.cadb_test_ratio,
            image_size=image_size,
            use_imagenet_norm=False,
        )
    if args.dataset == "cadb_elements":
        return build_cadb_elements_multilabel_dataloaders(
            cadb_root=args.cadb_root or (args.data_dir / "CADB_Dataset"),
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            val_ratio=args.val_ratio,
            image_size=image_size,
            use_imagenet_norm=False,
        )
    raise ValueError(f"Unsupported dataset for ResNet18 scratch: {args.dataset}")


def build_resnet18_imagenet_dataloaders(args):
    image_size = args.image_size or 224
    if args.dataset == "cadb_orientation":
        return build_cadb_orientation_dataloaders(
            cadb_root=args.cadb_root or (args.data_dir / "CADB_Dataset"),
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.cadb_test_ratio,
            image_size=image_size,
            label_mode=args.cadb_label_mode,
            balance_mode=args.cadb_balance_mode,
            use_imagenet_norm=True,
        )
    if args.dataset == "cadb_scene":
        return build_cadb_scene_dataloaders(
            cadb_root=args.cadb_root or (args.data_dir / "CADB_Dataset"),
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            num_workers=args.num_workers,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.cadb_test_ratio,
            image_size=image_size,
            use_imagenet_norm=True,
        )
    if args.dataset == "cadb_elements":
        raise ValueError("resnet18_imagenet is not enabled for the CADB multi-label elements task yet.")
    if args.dataset in {"synthetic_orientation", "synthetic_orientation_clean", "synthetic_orientation_hard", "synthetic_row_code", "synthetic_col_code"}:
        raise ValueError("resnet18_imagenet currently supports only cifar10, cadb_orientation, and cadb_scene.")
    if args.dataset != "cifar10":
        raise ValueError("resnet18_imagenet currently supports only cifar10, cadb_orientation, and cadb_scene.")
    return build_resnet_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        test_subset=args.test_subset,
        num_workers=args.num_workers,
        seed=args.seed,
        val_ratio=args.val_ratio,
        image_size=image_size,
        use_imagenet_norm=True,
    )


register_experiment(
    ExperimentSpec(
        model_name="vit_baseline",
        architecture="vit",
        variant="baseline",
        plot_title_prefix="ViT Baseline (No Pos)",
        defaults={"batch_size": 128, "lr": 3e-4, "weight_decay": 0.05},
        build_model=build_vit_baseline_model,
        build_dataloaders=build_vit_family_dataloaders,
        extra_summary_fields={"position_encoding": "none"},
    )
)

register_experiment(
    ExperimentSpec(
        model_name="vit_learnable_position",
        architecture="vit",
        variant="learnable_position",
        plot_title_prefix="ViT Learnable Position",
        defaults={"batch_size": 128, "lr": 3e-4, "weight_decay": 0.05},
        build_model=build_vit_learnable_position_model,
        build_dataloaders=build_vit_family_dataloaders,
        extra_summary_fields={"position_encoding": "absolute"},
    )
)

register_experiment(
    ExperimentSpec(
        model_name="vit_rope",
        architecture="vit",
        variant="rope",
        plot_title_prefix="ViT RoPE",
        defaults={"batch_size": 128, "lr": 3e-4, "weight_decay": 0.05},
        build_model=build_vit_rope_model,
        build_dataloaders=build_vit_family_dataloaders,
        extra_summary_fields={"position_encoding": "rope"},
    )
)

register_experiment(
    ExperimentSpec(
        model_name="vit_row_sinusoidal",
        architecture="vit",
        variant="row_sinusoidal",
        plot_title_prefix="ViT Row-wise Sinusoidal",
        defaults={"batch_size": 128, "lr": 3e-4, "weight_decay": 0.05},
        build_model=build_vit_row_sinusoidal_model,
        build_dataloaders=build_vit_family_dataloaders,
        extra_summary_fields={"position_encoding": "row_sinusoidal"},
    )
)

register_experiment(
    ExperimentSpec(
        model_name="vit_col_sinusoidal",
        architecture="vit",
        variant="col_sinusoidal",
        plot_title_prefix="ViT Column-wise Sinusoidal",
        defaults={"batch_size": 128, "lr": 3e-4, "weight_decay": 0.05},
        build_model=build_vit_col_sinusoidal_model,
        build_dataloaders=build_vit_family_dataloaders,
        extra_summary_fields={"position_encoding": "col_sinusoidal"},
    )
)

register_experiment(
    ExperimentSpec(
        model_name="vit_radial_sinusoidal",
        architecture="vit",
        variant="radial_sinusoidal",
        plot_title_prefix="ViT Radial Sinusoidal",
        defaults={"batch_size": 128, "lr": 3e-4, "weight_decay": 0.05},
        build_model=build_vit_radial_sinusoidal_model,
        build_dataloaders=build_vit_family_dataloaders,
        extra_summary_fields={"position_encoding": "radial_sinusoidal"},
    )
)

register_experiment(
    ExperimentSpec(
        model_name="vit_additive_sinusoidal",
        architecture="vit",
        variant="additive_sinusoidal",
        plot_title_prefix="ViT Additive Sinusoidal",
        defaults={"batch_size": 128, "lr": 3e-4, "weight_decay": 0.05},
        build_model=build_vit_additive_sinusoidal_model,
        build_dataloaders=build_vit_family_dataloaders,
        extra_summary_fields={"position_encoding": "additive_sinusoidal"},
    )
)

register_experiment(
    ExperimentSpec(
        model_name="vit_additive_sinusoidal_shifted",
        architecture="vit",
        variant="additive_sinusoidal_shifted",
        plot_title_prefix="ViT Additive Sinusoidal Shifted",
        defaults={"batch_size": 128, "lr": 3e-4, "weight_decay": 0.05},
        build_model=build_vit_additive_sinusoidal_shifted_model,
        build_dataloaders=build_vit_family_dataloaders,
        extra_summary_fields={"position_encoding": "additive_sinusoidal_shifted"},
    )
)

register_experiment(
    ExperimentSpec(
        model_name="vit_multiplicative_sinusoidal",
        architecture="vit",
        variant="multiplicative_sinusoidal",
        plot_title_prefix="ViT Multiplicative Sinusoidal",
        defaults={"batch_size": 128, "lr": 3e-4, "weight_decay": 0.05},
        build_model=build_vit_multiplicative_sinusoidal_model,
        build_dataloaders=build_vit_family_dataloaders,
        extra_summary_fields={"position_encoding": "multiplicative_sinusoidal"},
    )
)

register_experiment(
    ExperimentSpec(
        model_name="vit_multiplicative_sinusoidal_shifted",
        architecture="vit",
        variant="multiplicative_sinusoidal_shifted",
        plot_title_prefix="ViT Multiplicative Sinusoidal Shifted",
        defaults={"batch_size": 128, "lr": 3e-4, "weight_decay": 0.05},
        build_model=build_vit_multiplicative_sinusoidal_shifted_model,
        build_dataloaders=build_vit_family_dataloaders,
        extra_summary_fields={"position_encoding": "multiplicative_sinusoidal_shifted"},
    )
)

register_experiment(
    ExperimentSpec(
        model_name="vit_squared_multiplicative_sinusoidal",
        architecture="vit",
        variant="squared_multiplicative_sinusoidal",
        plot_title_prefix="ViT Squared Multiplicative Sinusoidal",
        defaults={"batch_size": 128, "lr": 3e-4, "weight_decay": 0.05},
        build_model=build_vit_squared_multiplicative_sinusoidal_model,
        build_dataloaders=build_vit_family_dataloaders,
        extra_summary_fields={"position_encoding": "squared_multiplicative_sinusoidal"},
    )
)

register_experiment(
    ExperimentSpec(
        model_name="vit_squared_multiplicative_sinusoidal_shifted",
        architecture="vit",
        variant="squared_multiplicative_sinusoidal_shifted",
        plot_title_prefix="ViT Squared Multiplicative Sinusoidal Shifted",
        defaults={"batch_size": 128, "lr": 3e-4, "weight_decay": 0.05},
        build_model=build_vit_squared_multiplicative_sinusoidal_shifted_model,
        build_dataloaders=build_vit_family_dataloaders,
        extra_summary_fields={"position_encoding": "squared_multiplicative_sinusoidal_shifted"},
    )
)

register_experiment(
    ExperimentSpec(
        model_name="vit_rope_2d",
        architecture="vit",
        variant="rope_2d",
        plot_title_prefix="ViT RoPE 2D",
        defaults={"batch_size": 128, "lr": 3e-4, "weight_decay": 0.05},
        build_model=build_vit_rope_2d_model,
        build_dataloaders=build_vit_family_dataloaders,
        extra_summary_fields={"position_encoding": "rope_2d"},
    )
)

register_experiment(
    ExperimentSpec(
        model_name="resnet18_scratch",
        architecture="resnet18",
        variant="scratch",
        plot_title_prefix="ResNet18 Scratch",
        defaults={"batch_size": 64, "lr": 1e-4, "weight_decay": 0.01},
        build_model=build_resnet18_scratch_model,
        build_dataloaders=build_resnet18_scratch_dataloaders,
        extra_summary_fields={"position_encoding": None},
    )
)

register_experiment(
    ExperimentSpec(
        model_name="resnet18_imagenet",
        architecture="resnet18",
        variant="imagenet",
        plot_title_prefix="ResNet18 ImageNet",
        defaults={"batch_size": 64, "lr": 1e-4, "weight_decay": 0.01},
        build_model=build_resnet18_imagenet_model,
        build_dataloaders=build_resnet18_imagenet_dataloaders,
        extra_summary_fields={"position_encoding": None},
    )
)


def resolve_experiment(args):
    spec = EXPERIMENT_REGISTRY[args.model]
    if args.batch_size is None:
        args.batch_size = spec.defaults["batch_size"]
    if args.lr is None:
        args.lr = spec.defaults["lr"]
    if args.weight_decay is None:
        args.weight_decay = spec.defaults["weight_decay"]
    return spec


def build_selected_experiment(args, spec):
    model, model_config = spec.build_model(args)
    train_loader, val_loader, test_loader = spec.build_dataloaders(args)
    metadata = {
        "architecture": spec.architecture,
        "variant": spec.variant,
        "plot_title_prefix": spec.plot_title_prefix,
        "task_type": "single_label",
        **spec.extra_summary_fields,
    }
    if args.dataset == "cadb_elements":
        metadata["task_type"] = "multilabel"
        metadata["label_names"] = list(CADB_ELEMENT_LABELS)
    return model, model_config, train_loader, val_loader, test_loader, metadata
