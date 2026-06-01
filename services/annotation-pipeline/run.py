#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from config import CONFIG


def cmd_preprocess(args):
    from preprocess import process_scene
    scene_path = Path(args.scene)
    if not scene_path.exists():
        print(f"[ERR] File not found: {scene_path}")
        sys.exit(1)
    saved = process_scene(scene_path, CONFIG)
    print(f"[OK] {len(saved)} tiles saved from {scene_path.name}")


def cmd_annotate(args):
    from ndvi_annotator import batch_generate
    saved = batch_generate(args.t1_scene, args.t2_scene, CONFIG)
    print(f"[OK] {len(saved)} masks auto-generated")


def cmd_visualize(args):
    from visualizer import render_annotation_ui
    render_annotation_ui()


def cmd_export(args):
    from exporter import export_all
    out = Path(args.output) if args.output else CONFIG.export_dir
    formats = args.formats.split(",") if args.formats else ["png"]
    result = export_all(out, CONFIG, formats=formats)
    print("[OK] Export results:")
    for k, v in result.items():
        if isinstance(v, dict):
            print(f"  {k}: {v.get('img', 0)} images")
        else:
            print(f"  {k}: {v}")


def cmd_status(args):
    cfg = CONFIG
    print(f"Raw scenes:     {len(list(cfg.raw_dir.glob('*.tif')))} files")
    print(f"Tiles:          {len(list(cfg.tiles_dir.glob('*.npz')))} files")
    print(f"Auto masks:     {len(list(cfg.masks_auto_dir.glob('*.npz')))} files")
    print(f"Refined masks:  {len(list(cfg.masks_refined_dir.glob('*.npz')))} files")
    print(f"Export:         {len(list(cfg.export_dir.glob('**/*.png')))} PNGs")


def main():
    parser = argparse.ArgumentParser(description="Deforest.id Annotation Pipeline")
    sub = parser.add_subparsers(dest="cmd")

    p_pre = sub.add_parser("preprocess", help="Tile & cloud-mask a scene GeoTIFF")
    p_pre.add_argument("scene", help="Path to GeoTIFF scene")

    p_ann = sub.add_parser("annotate", help="Auto-generate masks via NDVI change")
    p_ann.add_argument("t1_scene", help="Scene name (T1, earlier)")
    p_ann.add_argument("t2_scene", help="Scene name (T2, later)")

    sub.add_parser("visualize", help="Launch Streamlit refinement UI")

    p_exp = sub.add_parser("export", help="Export masks to training format")
    p_exp.add_argument("--output", "-o", default=None, help="Output directory")
    p_exp.add_argument("--formats", "-f", default="png",
                       help="Formats: png,geotiff (comma-sep)")

    sub.add_parser("status", help="Show pipeline status")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return

    CONFIG.ensure_dirs()
    globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    main()
