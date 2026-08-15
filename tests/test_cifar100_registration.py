import unittest
from argparse import Namespace

import torch

from datasets.cifar100_data import CIFAR100_MEAN, CIFAR100_STD
from models.registry import (
    DATASET_DEFAULT_IMAGE_SIZE,
    DATASET_NUM_CLASSES,
    SUPPORTED_DATASETS,
    build_vit_learnable_position_model,
)


class Cifar100RegistrationTests(unittest.TestCase):
    def test_dataset_metadata(self):
        self.assertIn("cifar100", SUPPORTED_DATASETS)
        self.assertEqual(DATASET_NUM_CLASSES["cifar100"], 100)
        self.assertEqual(DATASET_DEFAULT_IMAGE_SIZE["cifar100"], 32)
        self.assertEqual(len(CIFAR100_MEAN), 3)
        self.assertEqual(len(CIFAR100_STD), 3)

    def test_vit_classifier_has_one_hundred_outputs(self):
        args = Namespace(
            dataset="cifar100",
            image_size=None,
            embedding_dropout=0.0,
            attention_dropout=0.0,
            projection_dropout=0.0,
            mlp_dropout=0.0,
        )
        model, config = build_vit_learnable_position_model(args)
        logits = model(torch.zeros(2, 3, 32, 32))
        self.assertEqual(config["num_classes"], 100)
        self.assertEqual(tuple(logits.shape), (2, 100))


if __name__ == "__main__":
    unittest.main()
