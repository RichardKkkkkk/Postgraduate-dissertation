import unittest

import torch

from models.unfolding import (
    POSITION_ASSIGNMENT_MODES,
    UNFOLDING_MODES,
    build_patch_order,
    build_patch_position_mapping,
    order_fixed_positional_embedding,
)
from models.vit import PatchEmbedding


EXPECTED_3X3_ORDERS = {
    "normal_row": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "normal_col": [0, 3, 6, 1, 4, 7, 2, 5, 8],
    "proper_row": [0, 1, 2, 5, 4, 3, 6, 7, 8],
    "proper_col": [0, 3, 6, 7, 4, 1, 2, 5, 8],
}


class PatchPositionMappingTests(unittest.TestCase):
    def test_expected_patch_orders(self):
        for unfolding, expected in EXPECTED_3X3_ORDERS.items():
            with self.subTest(unfolding=unfolding):
                self.assertEqual(build_patch_order(3, unfolding).tolist(), expected)

    def test_mapping_distinguishes_physical_patch_and_pe_coordinate(self):
        for unfolding in UNFOLDING_MODES:
            with self.subTest(unfolding=unfolding):
                mapping = build_patch_position_mapping(3, unfolding)
                self.assertEqual(len(mapping), 9)
                for slot, record in enumerate(mapping):
                    physical_index = EXPECTED_3X3_ORDERS[unfolding][slot]
                    self.assertEqual(record["sequence_slot"], slot)
                    self.assertEqual(record["physical_patch_index"], physical_index)
                    self.assertEqual(
                        (record["physical_patch_row"], record["physical_patch_col"]),
                        divmod(physical_index, 3),
                    )
                    self.assertEqual(record["position_assignment"], "sequence_slot")
                    self.assertEqual(record["original_pe_index"], physical_index)
                    self.assertEqual(
                        (record["original_pe_row"], record["original_pe_col"]),
                        divmod(physical_index, 3),
                    )
                    self.assertEqual(record["assigned_pe_index"], slot)
                    self.assertEqual(
                        (record["assigned_pe_row"], record["assigned_pe_col"]),
                        divmod(slot, 3),
                    )

    def test_coordinate_aligned_mapping_preserves_physical_coordinates(self):
        self.assertIn("coordinate_aligned", POSITION_ASSIGNMENT_MODES)
        for unfolding in UNFOLDING_MODES:
            with self.subTest(unfolding=unfolding):
                mapping = build_patch_position_mapping(
                    3,
                    unfolding,
                    position_assignment="coordinate_aligned",
                )
                for record in mapping:
                    physical_coordinate = (
                        record["physical_patch_row"],
                        record["physical_patch_col"],
                    )
                    original_coordinate = (
                        record["original_pe_row"],
                        record["original_pe_col"],
                    )
                    assigned_coordinate = (
                        record["assigned_pe_row"],
                        record["assigned_pe_col"],
                    )
                    self.assertEqual(original_coordinate, physical_coordinate)
                    self.assertEqual(assigned_coordinate, physical_coordinate)

    def test_coordinate_aligned_positional_embedding_reorders_only_patch_part(self):
        # CLS is -1. Patch PE vectors carry their row-major physical index.
        full_pos_embed = torch.tensor([[[-1.0], [0.0], [1.0], [2.0], [3.0]]])
        order = build_patch_order(2, "normal_col")
        ordered = order_fixed_positional_embedding(
            full_pos_embed,
            order,
            position_assignment="coordinate_aligned",
        )
        self.assertEqual(ordered.squeeze(0).squeeze(-1).tolist(), [-1.0, 0.0, 2.0, 1.0, 3.0])

        legacy = order_fixed_positional_embedding(
            full_pos_embed,
            order,
            position_assignment="sequence_slot",
        )
        self.assertIs(legacy, full_pos_embed)

    def test_patch_embedding_applies_the_declared_order(self):
        # Each 1x1 physical patch contains its row-major index.  A unit-weight
        # projection makes the output token value equal to that patch index.
        image = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
        for unfolding, expected in EXPECTED_3X3_ORDERS.items():
            with self.subTest(unfolding=unfolding):
                layer = PatchEmbedding(
                    img_size=3,
                    patch_size=1,
                    in_channels=1,
                    embed_dim=1,
                    unfolding=unfolding,
                )
                with torch.no_grad():
                    layer.proj.weight.fill_(1.0)
                    layer.proj.bias.zero_()
                tokens = layer(image).squeeze(0).squeeze(-1)
                self.assertEqual(tokens.tolist(), [float(value) for value in expected])


if __name__ == "__main__":
    unittest.main()
