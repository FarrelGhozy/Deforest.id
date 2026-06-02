#!/usr/bin/env python3
"""
Deforest.id — GEE Training Data Export Pipeline
================================================
CLI untuk mengexport, men-tile, dan melabel data training U-Net
dari Google Earth Engine.

Usage:
    python -m src.run export --geojson aoi.json --prefix bovendigoel
    python -m src.run tile
    python -m src.run label-gfw
    python -m src.run split
    python -m src.run all --geojson aoi.json --prefix bovendigoel
"""
import argparse
import sys
from pathlib import Path


def cmd_export(args):
    from gee_export import authenticate, export_hl_scenes

    authenticate()
    date_ranges = [
        (args.t1_start, args.t1_end, "baseline"),
        (args.t2_start, args.t2_end, "deforest"),
    ]
    if args.t3_start and args.t3_end:
        date_ranges.append((args.t3_start, args.t3_end, "supplement"))

    tasks = export_hl_scenes(
        geojson_path=args.geojson,
        output_prefix=args.prefix,
        date_ranges=date_ranges,
        folder=args.folder,
    )

    if args.wait:
        from gee_export import monitor_tasks
        monitor_tasks(tasks)
    else:
        print(f"\n[INFO] {len(tasks)} task(s) started. "
              f"Run with --wait to monitor, or check GEE Tasks panel.")


def cmd_tile(args):
    from tile_unet import process_directory

    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    chips = process_directory(raw_dir)
    print(f"[DONE] {len(chips)} chips generated.")


def cmd_label_gfw(args):
    from label_gfw import generate_labels_from_chips

    chip_dir = Path(args.chip_dir) if args.chip_dir else None
    out_dir = Path(args.out_dir) if args.out_dir else None
    generate_labels_from_chips(chip_dir=chip_dir, output_dir=out_dir)


def cmd_split(args):
    from split_dataset import split_dataset

    chip_dir = Path(args.chip_dir) if args.chip_dir else None
    label_dir = Path(args.label_dir) if args.label_dir else None
    split_dataset(
        chip_dir=chip_dir,
        label_dir=label_dir,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )


def cmd_loss(args):
    from gee_export import authenticate, export_loss_batch, monitor_tasks

    authenticate()
    tasks = export_loss_batch(
        sample_dir=args.sample_dir,
        prefix=args.prefix,
        folder=args.folder,
    )

    if args.wait:
        monitor_tasks(tasks)
    else:
        print(f"\n[INFO] {len(tasks)} task(s) started. "
              f"Run with --wait to monitor, or check GEE Tasks panel.")


def cmd_all(args):
    print("=== Phase 1: GEE Export ===")
    cmd_export(args)

    print("\n=== Phase 2: Tiling ===")
    from tile_unet import process_directory
    process_directory(args.raw_dir)

    print("\n=== Phase 3: GFW Labeling ===")
    from label_gfw import generate_labels_from_chips
    generate_labels_from_chips()

    print("\n=== Phase 4: Dataset Split ===")
    from split_dataset import split_dataset
    stats = split_dataset(
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    print(f"\n{'='*40}")
    print(f"Pipeline complete: {stats['total']} total samples")
    print(f"  Train: {stats['train']}")
    print(f"  Val:   {stats['val']}")
    print(f"  Test:  {stats['test']}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Deforest.id — GEE Training Data Export Pipeline"
    )
    sub = p.add_subparsers(dest="command")

    # export
    ex = sub.add_parser("export", help="Export GeoTIFF composites from GEE")
    ex.add_argument("--geojson", required=True, help="Path to AOI GeoJSON")
    ex.add_argument("--prefix", required=True, help="Output filename prefix")
    ex.add_argument("--t1-start", default="2020-01-01")
    ex.add_argument("--t1-end", default="2020-12-31")
    ex.add_argument("--t2-start", default="2023-01-01")
    ex.add_argument("--t2-end", default="2023-12-31")
    ex.add_argument("--t3-start", default=None)
    ex.add_argument("--t3-end", default=None)
    ex.add_argument("--folder", default="deforest_training")
    ex.add_argument("--wait", action="store_true", help="Monitor until complete")

    # tile
    ti = sub.add_parser("tile", help="Tile GeoTIFFs into 64x64 chips")
    ti.add_argument("--raw-dir", default=None)

    # label-gfw
    lg = sub.add_parser("label-gfw", help="Generate weak labels from GFW Hansen")
    lg.add_argument("--chip-dir", default=None)
    lg.add_argument("--out-dir", default=None)

    # split
    sp = sub.add_parser("split", help="Split dataset into train/val/test")
    sp.add_argument("--chip-dir", default=None)
    sp.add_argument("--label-dir", default=None)
    sp.add_argument("--train-ratio", default="0.7")
    sp.add_argument("--val-ratio", default="0.2")
    sp.add_argument("--seed", default="42")

    # loss
    lo = sub.add_parser("loss", help="Export Hansen loss mask from GEE")
    lo.add_argument("--sample-dir", required=True, help="Directory with sample GeoJSONs")
    lo.add_argument("--prefix", default="hl_sample", help="Output filename prefix")
    lo.add_argument("--folder", default="deforest_training")
    lo.add_argument("--wait", action="store_true", help="Monitor until complete")

    # all
    al = sub.add_parser("all", help="Run full pipeline end-to-end")
    al.add_argument("--geojson", required=True)
    al.add_argument("--prefix", required=True)
    al.add_argument("--t1-start", default="2020-01-01")
    al.add_argument("--t1-end", default="2020-12-31")
    al.add_argument("--t2-start", default="2023-01-01")
    al.add_argument("--t2-end", default="2023-12-31")
    al.add_argument("--folder", default="deforest_training")
    al.add_argument("--raw-dir", default=None)
    al.add_argument("--train-ratio", default="0.7")
    al.add_argument("--val-ratio", default="0.2")
    al.add_argument("--seed", default="42")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "export": cmd_export,
        "tile": cmd_tile,
        "label-gfw": cmd_label_gfw,
        "split": cmd_split,
        "loss": cmd_loss,
        "all": cmd_all,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
