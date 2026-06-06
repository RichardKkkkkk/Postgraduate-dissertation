import argparse
from dataclasses import dataclass, field
from typing import Any, Callable

import torch.nn as nn
from torchvision import models

from cifar10_data import build_resnet_dataloaders, build_vit_dataloaders
from vit import ViT
from vit_rope import ViTRoPE


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


def register_experiment(spec: ExperimentSpec):
    if spec.model_name in EXPERIMENT_REGISTRY:
        raise ValueError(f"Duplicate experiment model name: {spec.model_name}")
    EXPERIMENT_REGISTRY[spec.model_name] = spec


def build_vit_model_config(args):
    return {
        "img_size": 32,
        "patch_size": 4,
        "in_channels": 3,
        "num_classes": 10,
        "embed_dim": 128,
        "num_blocks": 4,
        "num_heads": 4,
        "mlp_hidden_dim": 512,
        "embedding_dropout": args.embedding_dropout,
        "attention_dropout": args.attention_dropout,
        "projection_dropout": args.projection_dropout,
        "mlp_dropout": args.mlp_dropout,
    }


def build_vit_baseline_model(args):
    model_config = build_vit_model_config(args)
    return ViT(**model_config), model_config


def build_vit_rope_model(args):
    model_config = build_vit_model_config(args)
    model_config["rope_base"] = args.rope_base
    return ViTRoPE(**model_config), model_config


def build_vit_family_dataloaders(args):
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


def build_resnet18_model(weights_name):
    if weights_name == "imagenet":
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def build_resnet18_scratch_model(args):
    image_size = args.image_size or 32
    model = build_resnet18_model("none")
    model_config = {
        "architecture": "resnet18",
        "variant": "scratch",
        "weights": "none",
        "image_size": image_size,
        "num_classes": 10,
    }
    return model, model_config


def build_resnet18_imagenet_model(args):
    image_size = args.image_size or 224
    model = build_resnet18_model("imagenet")
    model_config = {
        "architecture": "resnet18",
        "variant": "imagenet",
        "weights": "imagenet",
        "image_size": image_size,
        "num_classes": 10,
    }
    return model, model_config


def build_resnet18_scratch_dataloaders(args):
    image_size = args.image_size or 32
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


def build_resnet18_imagenet_dataloaders(args):
    image_size = args.image_size or 224
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
        plot_title_prefix="CIFAR-10 ViT Baseline",
        defaults={"batch_size": 128, "lr": 3e-4, "weight_decay": 0.05},
        build_model=build_vit_baseline_model,
        build_dataloaders=build_vit_family_dataloaders,
        extra_summary_fields={"position_encoding": "absolute"},
    )
)

register_experiment(
    ExperimentSpec(
        model_name="vit_rope",
        architecture="vit",
        variant="rope",
        plot_title_prefix="CIFAR-10 ViT RoPE",
        defaults={"batch_size": 128, "lr": 3e-4, "weight_decay": 0.05},
        build_model=build_vit_rope_model,
        build_dataloaders=build_vit_family_dataloaders,
        extra_summary_fields={"position_encoding": "rope"},
    )
)

register_experiment(
    ExperimentSpec(
        model_name="resnet18_scratch",
        architecture="resnet18",
        variant="scratch",
        plot_title_prefix="CIFAR-10 ResNet18",
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
        plot_title_prefix="CIFAR-10 ResNet18",
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
        **spec.extra_summary_fields,
    }
    return model, model_config, train_loader, val_loader, test_loader, metadata
