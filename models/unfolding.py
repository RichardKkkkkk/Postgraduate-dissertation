import torch


UNFOLDING_MODES = ("normal_row", "normal_col", "proper_row", "proper_col")


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
