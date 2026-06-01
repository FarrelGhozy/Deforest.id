import streamlit as st
import numpy as np
from pathlib import Path
from config import CONFIG


def load_mask_npz(path: Path):
    return np.load(path)


def render_annotation_ui():
    st.set_page_config(layout="wide", page_title="Annotasi Deforest.id")
    st.title("🧠 Annotasi Deforest.id — Refine Mask Segmentasi")

    cfg = CONFIG

    mask_files = sorted(cfg.masks_auto_dir.glob("*_mask.npz"))
    if not mask_files:
        st.warning("Belum ada mask auto. Jalankan ndvi_annotator dulu.")
        return

    if "selected_idx" not in st.session_state:
        st.session_state.selected_idx = 0
        st.session_state.refined = {}
        st.session_state.loaded = None

    col_ctl, col_main = st.columns([1, 3])

    with col_ctl:
        file_names = [f.name for f in mask_files]
        selected_name = st.selectbox("Pilih tile", file_names,
                                     index=st.session_state.selected_idx)
        st.session_state.selected_idx = file_names.index(selected_name)

        st.divider()
        st.subheader("Auto Mask Stats")
        data = load_mask_npz(mask_files[st.session_state.selected_idx])
        mask = data["mask"]
        darea = int(mask.sum())
        dpct = darea / mask.size * 100
        st.metric("Area terdeteksi (px)", darea)
        st.metric("% area", f"{dpct:.1f}")

        st.divider()
        st.subheader("Refinement Tools")
        brush_size = st.slider("Brush size", 5, 100, 25)
        tool = st.radio("Tool", ["add", "erase"], horizontal=True)

        st.divider()
        st.subheader("Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✓ Confirm mask", use_container_width=True):
                save_refined(data, mask_files[st.session_state.selected_idx])
                st.success("Tersimpan!")
        with col2:
            if st.button("Skip tile", use_container_width=True):
                st.session_state.selected_idx = min(
                    st.session_state.selected_idx + 1, len(mask_files) - 1
                )
                st.rerun()

        st.divider()
        progress = len(list(cfg.masks_refined_dir.glob("*.npz")))
        total = len(mask_files)
        st.progress(progress / max(total, 1), text=f"{progress}/{total} refined")

    with col_main:
        data = load_mask_npz(mask_files[st.session_state.selected_idx])
        rgb = data["rgb"]
        mask = data["mask"].copy()
        tile_name = mask_files[st.session_state.selected_idx].name

        rgb_display = np.transpose(rgb, (1, 2, 0)).astype(np.uint8)
        overlay = make_overlay(rgb_display, mask)

        col_img, col_overlay = st.columns(2)
        with col_img:
            st.image(rgb_display, caption=f"Citra RGB — {tile_name}",
                     use_container_width=True)
        with col_overlay:
            st.image(overlay, caption="Overlay auto mask",
                     use_container_width=True)

        st.subheader("Refine Mask (click to add/erase)")
        st.caption("Klik kiri untuk add area deforestasi, klik kanan untuk erase")

        canvas_result = st.image(overlay, use_container_width=True)


def make_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    mask_bool = mask > 0
    overlay[..., 0][mask_bool] = np.clip(
        overlay[..., 0][mask_bool] * 0.3 + 200 * 0.7, 0, 255
    )
    return overlay


def save_refined(data, auto_path: Path):
    cfg = CONFIG
    mask = data["mask"]
    out_path = cfg.masks_refined_dir / auto_path.name

    rgb = data["rgb"]
    np.savez_compressed(
        out_path,
        mask=mask,
        rgb=rgb,
        ndvi_t1=data["ndvi_t1"],
        ndvi_t2=data["ndvi_t2"],
        transform=data["transform"],
        bounds=data["bounds"],
        scene_t1=str(data["scene_t1"]),
        scene_t2=str(data["scene_t2"]),
        refined=True,
    )


if __name__ == "__main__":
    render_annotation_ui()
