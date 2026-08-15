import torch


UNFOLDING_MODES = ("normal_row", "normal_col", "proper_row", "proper_col")
POSITION_ASSIGNMENT_MODES = ("sequence_slot", "coordinate_aligned")


def build_patch_order(grid_size, unfolding="normal_row"):
    grid = torch.arange(grid_size * grid_size).reshape(grid_size, grid_size)

    if unfolding == "normal_row":
        return grid.reshape(-1)
    if unfolding == "normal_col":
        return grid.t().reshape(-1)
    if unfolding == "proper_row":
        rows = [grid[row_idx].flip(0) if row_idx % 2 else grid[row_idx] for row_idx in range(grid_size)]
        return torch.cat(rows)
    if unfolding == "proper_col":
        cols = [grid[:, col_idx].flip(0) if col_idx % 2 else grid[:, col_idx] for col_idx in range(grid_size)]
        return torch.cat(cols)

    raise ValueError(f"Unsupported unfolding mode: {unfolding}")


def validate_position_assignment(position_assignment):
    if position_assignment not in POSITION_ASSIGNMENT_MODES:
        raise ValueError(
            f"Unsupported position assignment: {position_assignment}. "
            f"Expected one of {POSITION_ASSIGNMENT_MODES}."
        )
    return position_assignment


def order_fixed_positional_embedding(
    full_pos_embed,
    patch_order,
    position_assignment="sequence_slot",
):
    """Return fixed PE in token order while leaving the CLS entry untouched."""

    validate_position_assignment(position_assignment)
    if position_assignment == "sequence_slot":
        return full_pos_embed
    if full_pos_embed.ndim != 3 or full_pos_embed.shape[1] != patch_order.numel() + 1:
        raise ValueError(
            "Expected full_pos_embed shape (1, num_patches + 1, embed_dim) "
            f"for {patch_order.numel()} patches, got {tuple(full_pos_embed.shape)}"
        )
    cls_pos_embed = full_pos_embed[:, :1]
    patch_pos_embed = full_pos_embed[:, 1:].index_select(
        1, patch_order.to(device=full_pos_embed.device)
    )
    return torch.cat((cls_pos_embed, patch_pos_embed), dim=1)


def build_patch_position_mapping(
    grid_size,
    unfolding="normal_row",
    position_assignment="sequence_slot",
):
    """Describe physical patch, sequence slot, and assigned PE coordinates.

    ``sequence_slot`` is the historical behaviour: the reordered physical patch
    in slot ``s`` receives row-major PE coordinate ``s``.  In
    ``coordinate_aligned`` mode, fixed PE follows the same patch permutation.
    """

    validate_position_assignment(position_assignment)
    order = build_patch_order(grid_size, unfolding).tolist()
    mapping = []
    for sequence_slot, physical_index in enumerate(order):
        assigned_pe_index = (
            physical_index if position_assignment == "coordinate_aligned" else sequence_slot
        )
        mapping.append(
            {
                "unfolding": unfolding,
                "position_assignment": position_assignment,
                "sequence_slot": sequence_slot,
                "physical_patch_index": physical_index,
                "physical_patch_row": physical_index // grid_size,
                "physical_patch_col": physical_index % grid_size,
                "original_pe_index": physical_index,
                "original_pe_row": physical_index // grid_size,
                "original_pe_col": physical_index % grid_size,
                "assigned_pe_index": assigned_pe_index,
                "assigned_pe_row": assigned_pe_index // grid_size,
                "assigned_pe_col": assigned_pe_index % grid_size,
            }
        )
    return mapping
