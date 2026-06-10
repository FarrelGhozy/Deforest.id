import ee
import json
from pathlib import Path
from typing import Optional
from config import EXPORT


def authenticate(project: str = None, credentials: str = None):
    project = project or EXPORT.gee_project
    credentials = credentials or EXPORT.gee_credentials
    try:
        ee.Initialize(project=project)
    except Exception:
        cred_path = Path(credentials)
        if cred_path.exists():
            ee.ServiceAccountCredentials(
                email=None, key_file=str(cred_path), project=project
            )
            ee.Initialize(project=project)
        else:
            ee.Authenticate()
            ee.Initialize(project=project)


def load_aoi(geojson_path: str) -> ee.Geometry:
    with open(geojson_path) as f:
        fc = json.load(f)
    if fc["type"] == "FeatureCollection":
        coords = fc["features"][0]["geometry"]["coordinates"]
        geom_type = fc["features"][0]["geometry"]["type"]
    else:
        coords = fc["geometry"]["coordinates"]
        geom_type = fc["geometry"]["type"]

    if geom_type == "Polygon":
        return ee.Geometry.Polygon(coords)
    elif geom_type == "MultiPolygon":
        return ee.Geometry.MultiPolygon(coords)
    raise ValueError(f"Unsupported geometry type: {geom_type}")


def build_composite(
    aoi: ee.Geometry,
    date_start: str,
    date_end: str,
    cloud_filter: int = None,
    months: int = None,
) -> ee.Image:
    cloud_filter = cloud_filter or EXPORT.cloud_percent_filter
    months = months or EXPORT.composite_months

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_filter))
    )

    def mask_clouds(img):
        qa = img.select('QA60')
        cloud = qa.bitwiseAnd(1 << 10)
        cirrus = qa.bitwiseAnd(1 << 11)
        clear = cloud.eq(0).And(cirrus.eq(0))
        return img.updateMask(clear)

    cleaned = collection.map(mask_clouds)
    clear_count = cleaned.select('B2').count().rename('CLEAR_COUNT')
    composite = cleaned.median().clip(aoi)
    clear_count = clear_count.clip(aoi)

    return composite.addBands(clear_count).select(list(EXPORT.band_order))


def export_composite(
    composite: ee.Image,
    description: str,
    aoi: ee.Geometry,
    folder: str = None,
    scale: int = None,
    crs: str = None,
    max_pixels: int = None,
) -> ee.batch.Task:
    folder = folder or EXPORT.export_folder
    scale = scale or EXPORT.scale
    crs = crs or EXPORT.crs
    max_pixels = max_pixels or EXPORT.max_pixels

    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=aoi,
        scale=scale,
        crs=crs,
        maxPixels=max_pixels,
    )
    task.start()
    return task


def export_hl_scenes(
    geojson_path: str,
    output_prefix: str,
    date_ranges: list,
    folder: str = None,
):
    aoi = load_aoi(geojson_path)
    tasks = []
    for date_start, date_end, label in date_ranges:
        desc = f"{output_prefix}_{label}"
        composite = build_composite(aoi, date_start, date_end)
        task = export_composite(composite, desc, aoi, folder=folder)
        tasks.append({"task": task, "description": desc, "label": label})
        print(f"[EXPORT] Started: {desc}")
    return tasks


def monitor_tasks(tasks: list, check_interval: int = 30):
    import time

    pending = {t["description"]: t for t in tasks}
    while pending:
        for desc in list(pending.keys()):
            status = pending[desc]["task"].status()["state"]
            if status in ("COMPLETED", "FAILED", "CANCELLED"):
                print(f"[{status}] {desc}")
                del pending[desc]
        if pending:
            time.sleep(check_interval)
    print("[DONE] All exports finished.")


def build_loss_image(aoi: ee.Geometry) -> ee.Image:
    hansen = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
    loss = hansen.select("loss")
    return loss.clip(aoi).updateMask(loss.gt(0)).unmask(0)


def export_loss_batch(
    sample_dir: str,
    prefix: str = "hl_sample",
    folder: str = None,
    scale: int = 30,
    crs: str = None,
):
    sample_dir = Path(sample_dir)
    tasks = []
    for geojson_path in sorted(sample_dir.glob("*.geojson")):
        stem = geojson_path.stem  # e.g. sample_1
        loss_img = build_loss_image(load_aoi(str(geojson_path)))
        desc = f"{prefix}_{stem}_loss"
        task = export_composite(loss_img, desc, load_aoi(str(geojson_path)),
                                folder=folder, scale=scale, crs=crs)
        tasks.append({"task": task, "description": desc, "label": "loss"})
        print(f"[EXPORT] Started loss: {desc}")
    return tasks
