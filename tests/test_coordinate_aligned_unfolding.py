import unittest

import torch

from models.unfolding import UNFOLDING_MODES
from models.vit_axis_sinusoidal import (
    ViTAdditiveSinusoidal,
    ViTColSinusoidal,
    ViTMultiplicativeSinusoidal,
    ViTRadialSinusoidal,
    ViTRowSinusoidal,
)


FIXED_PE_CLASSES = (
    ViTRowSinusoidal,
    ViTColSinusoidal,
    ViTAdditiveSinusoidal,
    ViTMultiplicativeSinusoidal,
    ViTRadialSinusoidal,
)


class CoordinateAlignedUnfoldingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.input = torch.randn(2, 3, 8, 8)
        self.kwargs = {
            "img_size": 8,
            "patch_size": 2,
            "in_channels": 3,
            "embed_dim": 16,
            "num_heads": 4,
            "mlp_hidden_dim": 32,
            "num_blocks": 2,
            "num_classes": 3,
            "embedding_dropout": 0.0,
            "attention_dropout": 0.0,
            "projection_dropout": 0.0,
            "mlp_dropout": 0.0,
            "position_assignment": "coordinate_aligned",
        }

    def test_four_unfoldings_are_forward_equivalent_for_each_fixed_pe(self):
        for model_class in FIXED_PE_CLASSES:
            with self.subTest(model_class=model_class.__name__):
                reference = model_class(unfolding="normal_row", **self.kwargs).eval()
                reference_logits = reference(self.input)
                for unfolding in UNFOLDING_MODES:
                    candidate = model_class(unfolding=unfolding, **self.kwargs).eval()
                    candidate.load_state_dict(reference.state_dict(), strict=True)
                    logits = candidate(self.input)
                    max_abs_error = (logits - reference_logits).abs().max().item()
                    self.assertLessEqual(
                        max_abs_error,
                        1e-5,
                        msg=(
                            f"{model_class.__name__}/{unfolding} max absolute "
                            f"logit error was {max_abs_error:.3e}"
                        ),
                    )

    def test_default_assignment_remains_legacy_sequence_slot(self):
        default_model = ViTMultiplicativeSinusoidal(
            unfolding="normal_col",
            **{
                key: value
                for key, value in self.kwargs.items()
                if key != "position_assignment"
            },
        )
        explicit_legacy_model = ViTMultiplicativeSinusoidal(
            unfolding="normal_col",
            **{
                **self.kwargs,
                "position_assignment": "sequence_slot",
            },
        )
        explicit_legacy_model.load_state_dict(default_model.state_dict(), strict=True)
        default_model.eval()
        explicit_legacy_model.eval()
        self.assertEqual(default_model.position_assignment, "sequence_slot")
        self.assertTrue(
            torch.equal(default_model(self.input), explicit_legacy_model(self.input))
        )

    def test_all_fixed_pe_and_unfolding_combinations_smoke(self):
        for model_class in FIXED_PE_CLASSES:
            for unfolding in UNFOLDING_MODES:
                with self.subTest(model_class=model_class.__name__, unfolding=unfolding):
                    model = model_class(unfolding=unfolding, **self.kwargs).eval()
                    logits = model(self.input)
                    self.assertEqual(tuple(logits.shape), (2, 3))
                    self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
