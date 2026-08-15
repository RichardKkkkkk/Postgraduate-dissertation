import argparse
import csv
import json
from pathlib import Path

from models.unfolding import (
    POSITION_ASSIGNMENT_MODES,
    UNFOLDING_MODES,
    build_patch_position_mapping,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export physical-patch to fixed-PE coordinate mappings."
    )
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument(
        "--position-assignment",
        choices=POSITION_ASSIGNMENT_MODES,
        default="sequence_slot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/cifar10_final_vit_models_5seeds/reports/thesis_core"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for unfolding in UNFOLDING_MODES:
        records.extend(
            build_patch_position_mapping(
                args.grid_size,
                unfolding,
                position_assignment=args.position_assignment,
            )
        )

    csv_path = args.output_dir / f"patch_position_mapping_{args.position_assignment}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    manifest_path = args.output_dir / f"patch_position_mapping_{args.position_assignment}_manifest.json"
    manifest = {
        "grid_size": args.grid_size,
        "num_physical_patches": args.grid_size**2,
        "unfolding_modes": list(UNFOLDING_MODES),
        "position_assignment": args.position_assignment,
        "mapping_definition": (
            "physical patch order is selected by PatchEmbedding.patch_order; "
            + (
                "fixed patch PE follows the same permutation and remains aligned "
                "to the physical patch coordinate"
                if args.position_assignment == "coordinate_aligned"
                else "the fixed positional buffer remains indexed by row-major sequence slot"
            )
        ),
        "csv": str(csv_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(csv_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
