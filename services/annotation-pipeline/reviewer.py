"""Streamlit app for reviewing and refining pseudo-labels.

Usage:
    uv run streamlit run services/annotation-pipeline/reviewer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import numpy as np
import cv2
import json
from datetime import datetime

st.set_page_config(
    page_title="Deforest.id — Mask Reviewer",
    page_icon=":evergreen_tree:",
    layout="wide",
)

# ── Config ──────────────────────────────────────────────────

MASKS_DIR = Path("data/training/unet/labels_ndvi")
CHIPS_DIR = Path("data/training/unet/chips")
REVIEW_FILE = Path("data/training/unet/review_progress.json")


def load_mask_list(masks_dir: Path):
    return sorted(masks_dir.glob("*_mask.npz"))


def load_mask_data(path: Path):
    data = np.load(path)
    mask = data["mask"]
    rgb = np.transpose(data["rgb"], (1, 2, 0)).astype(np.uint8)
    ndvi_t1 = data.get("ndvi_t1", data.get("ndvi_baseline", None))
    ndvi_t2 = data.get("ndvi_t2", data.get("ndvi_deforest", None))
    return {
        "mask": mask,
        "rgb": rgb,
        "ndvi_t1": ndvi_t1,
        "ndvi_t2": ndvi_t2,
        "ndvi_change": ndvi_t2 - ndvi_t1 if ndvi_t1 is not None and ndvi_t2 is not None else None,
        "bounds": data.get("bounds", None),
        "scene_t1": str(data.get("scene_t1", data.get("scene_baseline", b""))),
        "scene_t2": str(data.get("scene_t2", data.get("scene_deforest", b""))),
        "path": str(path),
    }


def make_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    overlay = rgb.copy().astype(np.float32)
    mask_bool = mask > 0
    overlay[..., 0][mask_bool] = overlay[..., 0][mask_bool] * (1 - alpha) + 220 * alpha
    overlay[..., 1][mask_bool] = overlay[..., 1][mask_bool] * (1 - alpha)
    overlay[..., 2][mask_bool] = overlay[..., 2][mask_bool] * (1 - alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def compute_ndvi_change(ndvi_t1: np.ndarray, ndvi_t2: np.ndarray, threshold: float):
    return ((ndvi_t2 - ndvi_t1) < threshold).astype(np.uint8)


def apply_morphology(mask: np.ndarray, kernel_size: int, min_area: int = 32) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            cleaned[labels == i] = 0
    return cleaned


def load_review_state():
    if REVIEW_FILE.exists():
        return json.loads(REVIEW_FILE.read_text())
    return {"status": {}, "notes": {}}


def save_review_state(state: dict):
    REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_FILE.write_text(json.dumps(state, indent=2))


# ── Session State ───────────────────────────────────────────

if "mask_list" not in st.session_state:
    st.session_state.mask_list = load_mask_list(MASKS_DIR)
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "review_state" not in st.session_state:
    st.session_state.review_state = load_review_state()
if "edited_mask" not in st.session_state:
    st.session_state.edited_mask = None
if "comment" not in st.session_state:
    st.session_state.comment = ""

# ── UI ──────────────────────────────────────────────────────

st.title(":evergreen_tree: Deforest.id — Mask Reviewer")
st.caption("Review & refine pseudo-labels before U-Net training")

mask_list = st.session_state.mask_list
if not mask_list:
    st.warning(f"No `*_mask.npz` files found in `{MASKS_DIR}`")
    st.info("Run `generate_labels.py` or `ndvi_annotator.py` first")
    st.stop()

# Stats bar
review_state = st.session_state.review_state
total = len(mask_list)
statuses = review_state["status"]
reviewed = len(statuses)
accepted = sum(1 for v in statuses.values() if v == "accept")
rejected = sum(1 for v in statuses.values() if v == "reject")
needs_fix = sum(1 for v in statuses.values() if v == "needs_fix")

cols = st.columns(5)
cols[0].metric("Total", total)
cols[1].metric("Reviewed", reviewed, f"{reviewed/total*100:.0f}%" if total else "0%")
cols[2].metric(":white_check_mark: Accepted", accepted)
cols[3].metric(":x: Rejected", rejected)
cols[4].metric(":pencil: Needs Fix", needs_fix)

# Navigation
nav1, nav2, nav3, nav4, _ = st.columns([1, 1, 2, 2, 4])
with nav1:
    if st.button(":arrow_backward: Prev", use_container_width=True):
        st.session_state.idx = max(0, st.session_state.idx - 1)
        st.session_state.edited_mask = None
        st.rerun()
with nav2:
    if st.button("Next :arrow_forward:", use_container_width=True):
        st.session_state.idx = min(total - 1, st.session_state.idx + 1)
        st.session_state.edited_mask = None
        st.rerun()
with nav3:
    idx = st.number_input("", 0, total - 1, st.session_state.idx, label_visibility="collapsed")
    if idx != st.session_state.idx:
        st.session_state.idx = idx
        st.session_state.edited_mask = None
        st.rerun()
with nav4:
    st.text_input(
        "Filter mask name",
        placeholder="e.g. hl_sample_3",
        label_visibility="collapsed",
        key="filter_name",
    )

# ── Load current mask ───────────────────────────────────────

current_idx = st.session_state.idx
mask_path = mask_list[current_idx]
data = load_mask_data(mask_path)
mask = data["mask"]
rgb = data["rgb"]
area_pct = mask.sum() / mask.size * 100
ndvi = data["ndvi_change"]

# Filter jump (if someone typed a filter name)
filter_val = st.session_state.get("filter_name", "")
if filter_val:
    filtered_indices = [
        i for i, p in enumerate(mask_list) if filter_val.lower() in p.stem.lower()
    ]
    if filtered_indices and current_idx not in filtered_indices:
        st.session_state.idx = filtered_indices[0]
        st.rerun()

# ── Image display ───────────────────────────────────────────

st.subheader(f"`{mask_path.stem[:60]}` — {current_idx+1}/{total}")

img_cols = st.columns(3)
with img_cols[0]:
    st.caption(":frame_photo: RGB Original")
    st.image(rgb, use_container_width=True)
    st.caption(f"Scene: {data['scene_t1'][:60]}")
with img_cols[1]:
    st.caption(":cinema: Mask Overlay")
    overlay = make_overlay(rgb, mask)
    st.image(overlay, use_container_width=True)
    st.caption(f"Deforest area: **{area_pct:.1f}%** | `{mask_path.name}`")
with img_cols[2]:
    st.caption(":bar_chart: NDVI Change (baseline :arrow_right: deforest)")
    fig_data = {
        "ndvi_mean_t1": float(data["ndvi_t1"].mean()) if data["ndvi_t1"] is not None else 0,
        "ndvi_mean_t2": float(data["ndvi_t2"].mean()) if data["ndvi_t2"] is not None else 0,
        "ndvi_delta": float(ndvi.mean()) if ndvi is not None else 0,
        "area_pct": area_pct,
    }
    st.json(fig_data)
    if ndvi is not None:
        ndvi_disp = (ndvi * 0.5 + 0.5).clip(0, 1)  # normalize for display
        st.image((ndvi_disp * 255).astype(np.uint8), use_container_width=True, clamp=True)
        st.caption(f":chart_with_downwards_trend: NDVI change (red = drop)")

# ── Threshold re-gen ────────────────────────────────────────

with st.expander(":control_knobs: Regenerate with different threshold", expanded=False):
    if ndvi is not None:
        thresh = st.slider("NDVI change threshold", -0.5, 0.0, -0.15, 0.01, format="%.2f")
        ks = st.slider("Morphology kernel", 1, 11, 5, 2)
        ma = st.number_input("Min area (px)", 16, 256, 64, 16)

        if st.button(":arrows_counterclockwise: Preview new mask", use_container_width=True):
            new_mask = compute_ndvi_change(data["ndvi_t1"], data["ndvi_t2"], thresh)
            new_mask = apply_morphology(new_mask, ks, ma)
            st.session_state.edited_mask = new_mask

            pre_a, pre_b = st.columns(2)
            with pre_a:
                st.image(make_overlay(rgb, new_mask), use_container_width=True)
                st.caption(f"New: {new_mask.sum()/new_mask.size*100:.1f}%")
            with pre_b:
                st.image(overlay, use_container_width=True)
                st.caption(f"Original: {area_pct:.1f}%")

            if st.button(":floppy_disk: Save this version as correction", use_container_width=True):
                np.savez_compressed(
                    mask_path,
                    mask=new_mask,
                    rgb=data.get("rgb_raw", np.transpose(rgb, (2, 0, 1))),
                    ndvi_t1=data["ndvi_t1"],
                    ndvi_t2=data["ndvi_t2"],
                    bounds=data["bounds"],
                    scene_t1=data["scene_t1"],
                    scene_t2=data["scene_t2"],
                )
                st.success(f"Saved! New area: {new_mask.sum()/new_mask.size*100:.1f}%")
                st.session_state.edited_mask = None
                st.rerun()

# ── Manual pixel correction ─────────────────────────────────

with st.expander(":pencil2: Manual pixel correction", expanded=False):
    st.info("""
        Klik pixel untuk toggle mask (0:arrow_right:1 atau 1:arrow_right:0).
        Gunakan brush size untuk area yang lebih besar.
        Klik **Apply** setelah selesai untuk menyimpan perubahan.
    """)

    current_mask = (
        st.session_state.edited_mask if st.session_state.edited_mask is not None else mask.copy()
    )

    brush_size = st.slider("Brush size (pixels)", 1, 15, 5, 2)

    # Render canvas with current mask
    rgb_list = rgb.tolist()
    mask_list_json = current_mask.tolist()

    canvas_html = f"""
    <div style="display:flex;gap:20px;flex-wrap:wrap;justify-content:center;">
      <div>
        <canvas id="c" width="320" height="320"
          style="border:1px solid #555;border-radius:8px;cursor:crosshair;image-rendering:pixelated;"></canvas>
        <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;">
          <button id="btn-paint" class="active" onclick="setMode('paint')"
            style="padding:4px 16px;border-radius:4px;border:1px solid #555;cursor:pointer;">Add</button>
          <button onclick="setMode('erase')"
            style="padding:4px 16px;border-radius:4px;border:1px solid #555;cursor:pointer;">Erase</button>
          <button onclick="setMode('toggle')"
            style="padding:4px 16px;border-radius:4px;border:1px solid #555;cursor:pointer;">Toggle</button>
          <button onclick="doReset()"
            style="padding:4px 16px;border-radius:4px;border:1px solid #555;cursor:pointer;">Reset</button>
        </div>
      </div>
      <canvas id="preview" width="320" height="320"
        style="border:1px solid #555;border-radius:8px;image-rendering:pixelated;"></canvas>
    </div>
    <div id="status"></div>
    <script>
    const canvas = document.getElementById('c');
    const preview = document.getElementById('preview');
    const ctx = canvas.getContext('2d');
    const pctx = preview.getContext('2d');
    const SCALE = 5;
    const rgbSrc = {json.dumps(rgb_list)};
    let mask = {json.dumps(mask_list_json)};
    let mode = 'paint';
    let drawing = false;
    const bs = {brush_size};

    function drawCanvas() {{
      const imgData = ctx.createImageData(320, 320);
      for (let y = 0; y < 64; y++) {{
        for (let x = 0; x < 64; x++) {{
          const r = rgbSrc[y][x][0], g = rgbSrc[y][x][1], b = rgbSrc[y][x][2];
          const m = mask[y][x] > 0;
          for (let dy = 0; dy < SCALE; dy++) {{
            for (let dx = 0; dx < SCALE; dx++) {{
              const px = ((y * SCALE + dy) * 320 + (x * SCALE + dx)) * 4;
              if (m) {{
                imgData.data[px] = r * 0.3 + 220 * 0.7;
                imgData.data[px+1] = g * 0.3;
                imgData.data[px+2] = b * 0.3;
              }} else {{
                imgData.data[px] = r; imgData.data[px+1] = g; imgData.data[px+2] = b;
              }}
              imgData.data[px+3] = 255;
            }}
          }}
        }}
      }}
      ctx.putImageData(imgData, 0, 0);

      const pData = pctx.createImageData(320, 320);
      for (let y = 0; y < 64; y++) {{
        for (let x = 0; x < 64; x++) {{
          const v = mask[y][x] > 0 ? 255 : 0;
          for (let dy = 0; dy < SCALE; dy++) {{
            for (let dx = 0; dx < SCALE; dx++) {{
              const px = ((y * SCALE + dy) * 320 + (x * SCALE + dx)) * 4;
              pData.data[px] = v; pData.data[px+1] = 0; pData.data[px+2] = 0;
              pData.data[px+3] = 255;
            }}
          }}
        }}
      }}
      pctx.putImageData(pData, 0, 0);
    }}

    function getPos(e) {{
      const rect = canvas.getBoundingClientRect();
      const x = Math.floor((e.clientX - rect.left) / (rect.width / 64));
      const y = Math.floor((e.clientY - rect.top) / (rect.height / 64));
      return [Math.max(0, Math.min(63, x)), Math.max(0, Math.min(63, y))];
    }}

    function applyAt(x, y) {{
      const half = Math.floor(bs / 2);
      for (let dy = -half; dy <= half; dy++) {{
        for (let dx = -half; dx <= half; dx++) {{
          const px = Math.max(0, Math.min(63, x + dx));
          const py = Math.max(0, Math.min(63, y + dy));
          if (mode === 'toggle') {{
            mask[py][px] = mask[py][px] > 0 ? 0 : 1;
          }} else if (mode === 'paint') {{
            mask[py][px] = 1;
          }} else {{
            mask[py][px] = 0;
          }}
        }}
      }}
      drawCanvas();
    }}

    canvas.addEventListener('mousedown', (e) => {{ e.preventDefault(); drawing = true; const [x,y] = getPos(e); applyAt(x,y); }});
    canvas.addEventListener('mousemove', (e) => {{ if (!drawing) return; const [x,y] = getPos(e); applyAt(x,y); }});
    canvas.addEventListener('mouseup', () => {{ drawing = false; }});
    canvas.addEventListener('mouseleave', () => {{ drawing = false; }});

    window.setMode = function(m) {{
      mode = m;
      document.querySelectorAll('#btn-paint, button:has-text("Erase"), button:has-text("Toggle")').forEach(b => b.className = '');
      // Simplification: just set mode
    }};

    window.doReset = function() {{
      mask = JSON.parse(JSON.stringify({json.dumps(mask_list_json)}));
      drawCanvas();
    }};

    drawCanvas();
    </script>
    """

    st.components.v1.html(canvas_html, height=400)

    col_a, col_b, col_c = st.columns([1, 1, 3])
    with col_a:
        if st.button(":floppy_disk: Apply & Save", use_container_width=True, type="primary"):
            st.info("Drawing correction applied. Use 'Save this version' above or manually overwrite.")
            # Store in session for now
            st.session_state.edited_mask = current_mask
            st.info("Mask saved to session. Use review buttons below to mark as 'Needs Fix'.")

    with col_b:
        if st.button(":wastebasket: Discard", use_container_width=True):
            st.session_state.edited_mask = None
            st.rerun()

    with col_c:
        # Show stats of edited mask
        if st.session_state.edited_mask is not None:
            new_area = st.session_state.edited_mask.sum() / st.session_state.edited_mask.size * 100
            diff_px = int(np.abs(st.session_state.edited_mask.astype(int) - mask.astype(int)).sum())
            st.caption(f"Edited: {new_area:.1f}% deforest | {diff_px} pixels changed")

# ── Review actions ──────────────────────────────────────────

st.divider()
st.subheader(":clipboard: Review Decision")

key = str(current_idx)
current_status = statuses.get(key)

ra, rb, rc, rd = st.columns([1, 1, 1, 2])
with ra:
    if st.button(":white_check_mark: Accept", use_container_width=True, type="primary"):
        review_state["status"][key] = "accept"
        review_state["notes"][key] = {
            "area_pct": area_pct,
            "edited": st.session_state.edited_mask is not None,
            "time": str(datetime.now()),
        }
        save_review_state(review_state)
        st.rerun()

with rb:
    if st.button(":x: Reject", use_container_width=True):
        review_state["status"][key] = "reject"
        review_state["notes"][key] = {"area_pct": area_pct, "time": str(datetime.now())}
        save_review_state(review_state)
        st.rerun()

with rc:
    if st.button(":pencil: Needs Fix", use_container_width=True):
        review_state["status"][key] = "needs_fix"
        review_state["notes"][key] = {
            "area_pct": area_pct,
            "edited": st.session_state.edited_mask is not None,
            "time": str(datetime.now()),
        }
        save_review_state(review_state)
        st.rerun()

with rd:
    if current_status:
        st.info(f"Status: **{current_status}**")
    else:
        st.info("Not yet reviewed")

# Comment per mask
new_comment = st.text_area(
    "Notes for this mask",
    value=review_state.get("notes", {}).get(key, {}).get("comment", ""),
    placeholder="e.g., false positive — cloud shadow detected as deforestation",
    label_visibility="collapsed",
)
if new_comment != st.session_state.get("_last_comment", ""):
    st.session_state._last_comment = new_comment
    if "notes" not in review_state:
        review_state["notes"] = {}
    if key not in review_state["notes"]:
        review_state["notes"][key] = {}
    review_state["notes"][key]["comment"] = new_comment
    save_review_state(review_state)

# ── Batch actions ───────────────────────────────────────────

st.divider()
batch_cols = st.columns([1, 1, 1, 4])
with batch_cols[0]:
    if st.button(":fast_forward: Skip to next unreviewed", use_container_width=True):
        for i in range(current_idx + 1, total):
            if str(i) not in review_state["status"]:
                st.session_state.idx = i
                st.session_state.edited_mask = None
                st.rerun()
        st.info("All reviewed!")

with batch_cols[1]:
    if st.button(":bar_chart: Export review results", use_container_width=True):
        export = {
            "total": total,
            "reviewed": reviewed,
            "accepted": accepted,
            "rejected": rejected,
            "needs_fix": needs_fix,
            "acceptance_rate": f"{accepted/(accepted+rejected)*100:.1f}%" if accepted + rejected > 0 else "N/A",
            "details": review_state,
        }
        Path("data/annotation/review_export.json").write_text(json.dumps(export, indent=2))
        st.success("Exported to `data/annotation/review_export.json`")
        st.json(export)

with batch_cols[2]:
    if st.button(":wastebasket: Export rejected images for re-labeling", use_container_width=True):
        rejected_indices = [int(k) for k, v in review_state["status"].items() if v == "reject"]
        rejected_list = [str(mask_list[i]) for i in rejected_indices]
        out = {"rejected_count": len(rejected_list), "files": rejected_list}
        Path("data/annotation/rejected_masks.json").write_text(json.dumps(out, indent=2))
        st.success(f"{len(rejected_list)} rejected masks listed in `rejected_masks.json`")
