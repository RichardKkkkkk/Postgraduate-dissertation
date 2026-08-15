import argparse
import csv
import json
from pathlib import Path

import torch

from models.unfolding import UNFOLDING_MODES, build_patch_position_mapping
from models.vit_axis_sinusoidal import (
    ViTAdditiveSinusoidal,
    ViTColSinusoidal,
    ViTMultiplicativeSinusoidal,
    ViTRadialSinusoidal,
    ViTRowSinusoidal,
)


FIXED_PE_MODELS = {
    "row_sinusoidal": ViTRowSinusoidal,
    "col_sinusoidal": ViTColSinusoidal,
    "additive_sinusoidal": ViTAdditiveSinusoidal,
    "multiplicative_sinusoidal": ViTMultiplicativeSinusoidal,
    "radial_sinusoidal": ViTRadialSinusoidal,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit coordinate-aligned patch/PE mappings and forward equivalence."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cifar10_coordinate_aligned_unfolding_5seeds/audit"),
    )
    parser.add_argument("--grid-size", type=int, default=8)
    return parser.parse_args()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_forward_equivalence():
    torch.manual_seed(123)
    input_tensor = torch.randn(2, 3, 32, 32)
    kwargs = {
        "img_size": 32,
        "patch_size": 4,
        "in_channels": 3,
        "embed_dim": 128,
        "num_heads": 4,
        "mlp_hidden_dim": 512,
        "num_blocks": 4,
        "num_classes": 10,
        "embedding_dropout": 0.0,
        "attention_dropout": 0.0,
        "projection_dropout": 0.0,
        "mlp_dropout": 0.0,
        "position_assignment": "coordinate_aligned",
    }
    rows = []
    with torch.no_grad():
        for pe_name, model_class in FIXED_PE_MODELS.items():
            reference = model_class(unfolding="normal_row", **kwargs).eval()
            reference_logits = reference(input_tensor)
            for unfolding in UNFOLDING_MODES:
                candidate = model_class(unfolding=unfolding, **kwargs).eval()
                candidate.load_state_dict(reference.state_dict(), strict=True)
                logits = candidate(input_tensor)
                max_abs_error = float((logits - reference_logits).abs().max().item())
                rows.append(
                    {
                        "position_encoding": pe_name,
                        "reference_unfolding": "normal_row",
                        "candidate_unfolding": unfolding,
                        "max_absolute_logit_error": f"{max_abs_error:.12e}",
                        "tolerance": "1e-5",
                        "status": "pass" if max_abs_error <= 1e-5 else "fail",
                    }
                )
    return rows


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mapping_rows = []
    for unfolding in UNFOLDING_MODES:
        mapping_rows.extend(
            build_patch_position_mapping(
                args.grid_size,
                unfolding,
                position_assignment="coordinate_aligned",
            )
        )
    mapping_csv = args.output_dir / "coordinate_aligned_patch_position_mapping.csv"
    write_csv(mapping_csv, mapping_rows)

    mapping_failures = [
        row
        for row in mapping_rows
        if (row["assigned_pe_row"], row["assigned_pe_col"])
        != (row["physical_patch_row"], row["physical_patch_col"])
    ]
    equivalence_rows = run_forward_equivalence()
    equivalence_csv = args.output_dir / "forward_equivalence.csv"
    write_csv(equivalence_csv, equivalence_rows)

    architecture_audit = {
        "global_self_attention": True,
        "attention_mask": None,
        "local_or_window_attention": False,
        "sequence_convolution": False,
        "position_dependent_mask": False,
        "token_pooling": "CLS token only after permutation-equivariant encoder blocks",
        "sequence_adjacent_operation": False,
        "extra_patch_indexed_tensors": ["fixed positional embedding, synchronously reordered"],
        "permutation_equivalence_expectation": (
            "Patch tokens and fixed PE are permuted together; shared token-wise LayerNorm/MLP "
            "and global unmasked self-attention preserve the CLS output up to floating-point error."
        ),
    }
    manifest = {
        "grid_size": args.grid_size,
        "position_assignment": "coordinate_aligned",
        "unfolding_modes": list(UNFOLDING_MODES),
        "fixed_pe_models": list(FIXED_PE_MODELS),
        "mapping_records": len(mapping_rows),
        "mapping_failures": len(mapping_failures),
        "forward_checks": len(equivalence_rows),
        "forward_failures": sum(row["status"] != "pass" for row in equivalence_rows),
        "maximum_observed_logit_error": max(
            float(row["max_absolute_logit_error"]) for row in equivalence_rows
        ),
        "architecture_audit": architecture_audit,
        "mapping_csv": str(mapping_csv),
        "forward_equivalence_csv": str(equivalence_csv),
    }
    manifest_path = args.output_dir / "coordinate_aligned_audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Mapping records: {len(mapping_rows)}; failures: {len(mapping_failures)}")
    print(
        f"Forward checks: {len(equivalence_rows)}; failures: {manifest['forward_failures']}; "
        f"maximum error: {manifest['maximum_observed_logit_error']:.3e}"
    )
    print(manifest_path)
    if mapping_failures or manifest["forward_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
