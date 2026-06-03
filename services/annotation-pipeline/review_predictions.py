"""Streamlit app to review U-Net predictions vs ground truth.

Usage:
    uv run streamlit run services/annotation-pipeline/review_predictions.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import numpy as np
import cv2
import json

st.set_page_config(
    page_title="Deforest.id — Prediction Reviewer",
    page_icon=":evergreen_tree:",
    layout="wide",
)

MANIFEST_PATH = Path("data/training/unet/manifest.json")
PREDICTIONS_PATH = Path("data/training/unet/predictions/predictions.npy")
IGNORE_LABEL = 255


def load_entries():
    with open(MANIFEST_PATH) as f:
        m = json.load(f)
    return m["test"]


def load_chip_rgb(entry):
    data = np.load(entry["image"])
    return np.transpose(data["rgb"], (1, 2, 0)).astype(np.uint8)


def load_gt_mask(entry):
    return np.load(entry["mask"])["mask"]


def upscale_nn(img, scale=7):
    h, w = img.shape[:2]
    interp = cv2.INTER_NEAREST
    if img.ndim == 2:
        return cv2.resize(img, (w * scale, h * scale), interpolation=interp)
    return cv2.resize(img, (w * scale, h * scale), interpolation=interp)


def make_overlay(rgb, mask, alpha=0.5):
    overlay = rgb.copy().astype(np.float32)
    o_def = mask == 1
    o_ign = mask == IGNORE_LABEL
    overlay[..., 2][o_def] = overlay[..., 2][o_def] * (1 - alpha) + 220 * alpha
    overlay[..., 1][o_def] *= (1 - alpha)
    overlay[..., 0][o_def] *= (1 - alpha)
    overlay[..., 2][o_ign] *= (1 - alpha)
    overlay[..., 1][o_ign] *= (1 - alpha)
    overlay[..., 0][o_ign] = overlay[..., 0][o_ign] * (1 - alpha) + 200 * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def compute_iou(pred, gt):
    valid = gt != IGNORE_LABEL
    if not valid.any():
        return 0.0
    inter = ((pred == 1) & (gt == 1) & valid).sum()
    union = ((pred == 1) | (gt == 1)) & valid
    return inter / (union.sum() + 1e-6)


if "entries" not in st.session_state:
    st.session_state.entries = load_entries()
if "preds" not in st.session_state:
    if PREDICTIONS_PATH.exists():
        st.session_state.preds = np.load(PREDICTIONS_PATH)
    else:
        st.session_state.preds = None
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "all_ious" not in st.session_state:
    st.session_state.all_ious = None

entries = st.session_state.entries
preds = st.session_state.preds
total = len(entries)

st.title(":evergreen_tree: Deforest.id — Prediction Reviewer")
st.caption("Compare U-Net predictions against ground truth on test set")

if preds is None:
    st.warning(f"No predictions at `{PREDICTIONS_PATH}`")
    st.info("Run `infer_unet.py` first")
    st.stop()

nav1, nav2, nav3, _ = st.columns([1, 1, 3, 5])
with nav1:
    if st.button(":arrow_backward: Prev", use_container_width=True):
        st.session_state.idx = max(0, st.session_state.idx - 1)
        st.rerun()
with nav2:
    if st.button("Next :arrow_forward:", use_container_width=True):
        st.session_state.idx = min(total - 1, st.session_state.idx + 1)
        st.rerun()
with nav3:
    idx_input = st.number_input("", 0, total - 1, st.session_state.idx, label_visibility="collapsed")
    if idx_input != st.session_state.idx:
        st.session_state.idx = idx_input
        st.rerun()

current_idx = st.session_state.idx
entry = entries[current_idx]
pred_mask = preds[current_idx]
gt_mask = load_gt_mask(entry)
rgb = load_chip_rgb(entry)

iou = compute_iou(pred_mask, gt_mask)
gt_area = (gt_mask == 1).sum() / gt_mask.size * 100
pred_area = (pred_mask == 1).sum() / pred_mask.size * 100

st.subheader(
    f"`{entry['stem'][:60]}` — {current_idx+1}/{total}  "
    f"|  IoU: **{iou:.3f}**  "
    f"|  GT: {gt_area:.1f}%  Pred: {pred_area:.1f}%"
)

overlay_gt = make_overlay(rgb, gt_mask)
overlay_pred = make_overlay(rgb, pred_mask)

error_map = np.zeros((64, 64, 3), dtype=np.uint8)
valid = gt_mask != IGNORE_LABEL
tp = (pred_mask == 1) & (gt_mask == 1) & valid
fp = (pred_mask == 1) & (gt_mask == 0) & valid
fn = (pred_mask == 0) & (gt_mask == 1) & valid
tn = (pred_mask == 0) & (gt_mask == 0) & valid
error_map[tp] = [0, 200, 0]
error_map[fp] = [200, 0, 0]
error_map[fn] = [0, 0, 200]
error_map[tn] = [30, 30, 30]

img_cols = st.columns(4)
with img_cols[0]:
    st.caption(":frame_photo: RGB")
    st.image(upscale_nn(rgb, 7), width=384)
with img_cols[1]:
    st.caption(":cinema: Ground Truth")
    st.image(upscale_nn(overlay_gt, 7), width=384)
    st.caption(f"Deforest: **{gt_area:.1f}%**")
with img_cols[2]:
    st.caption(":robot_face: Prediction")
    st.image(upscale_nn(overlay_pred, 7), width=384)
    st.caption(f"Deforest: **{pred_area:.1f}%** | IoU: **{iou:.3f}**")
with img_cols[3]:
    st.caption(":chart: Error Map (TP/FP/FN)")
    st.image(upscale_nn(error_map, 7), width=384)
    tp_pct = tp.sum() / valid.sum() * 100 if valid.any() else 0
    fp_pct = fp.sum() / valid.sum() * 100 if valid.any() else 0
    fn_pct = fn.sum() / valid.sum() * 100 if valid.any() else 0
    st.caption(f":green_circle: TP {tp_pct:.1f}%  :red_circle: FN {fn_pct:.1f}%  :blue_circle: FP {fp_pct:.1f}%")

st.divider()
st.subheader(":bar_chart: Test Set Overview")

if st.session_state.all_ious is None:
    all_ious = []
    for i in range(total):
        all_ious.append(compute_iou(preds[i], load_gt_mask(entries[i])))
    st.session_state.all_ious = np.array(all_ious)

all_ious = st.session_state.all_ious
valid_ious = all_ious[~np.isnan(all_ious)]
with open("data/training/unet/predictions/metrics.json") as f:
    metrics = json.load(f)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Mean IoU", f"{valid_ious.mean():.4f}")
m2.metric("Median IoU", f"{np.median(valid_ious):.4f}")
m3.metric("IoU > 0.5", f"{(valid_ious > 0.5).mean() * 100:.1f}%")
m4.metric("Dice", f"{metrics['Dice']:.4f}")

import altair as alt
import pandas as pd
hist_df = pd.DataFrame({"IoU": valid_ious})
chart = alt.Chart(hist_df).mark_bar(color="#2ecc71").encode(
    alt.X("IoU", bin=alt.Bin(maxbins=40), title="IoU"),
    alt.Y("count()", title="Samples"),
).properties(height=300)
st.altair_chart(chart, use_container_width=True)

st.caption(f"Total: {total} test samples | IoU ≤ 0: {(valid_ious <= 0).sum()} samples")
