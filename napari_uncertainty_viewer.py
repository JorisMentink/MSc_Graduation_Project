#!/usr/bin/env python
"""
Napari viewer for MRI image, nnUNet segmentation, and uncertainty map.

Usage:
    python napari_uncertainty_viewer.py [--subject_index N]

Example:
    python napari_uncertainty_viewer.py --subject_index 19
"""

import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import napari
from magicgui import magicgui
from scipy.ndimage import binary_erosion, distance_transform_edt, maximum_filter


def build_prediction_edge_mask(pred_mask: np.ndarray):
    """Build a 1-pixel prediction boundary mask slice-by-slice."""
    pred_bool = pred_mask.astype(bool)

    if pred_bool.ndim == 2:
        return pred_bool & ~binary_erosion(pred_bool, iterations=1)

    edge = np.zeros_like(pred_bool, dtype=bool)
    for z in range(pred_bool.shape[0]):
        sl = pred_bool[z]
        if sl.any():
            edge[z] = sl & ~binary_erosion(sl, iterations=1)
    return edge


def get_band_middle_inner_outer_2d(band_2d,seg_edge_2d,min_middle_half_width_px: float = 1.0,):
    
    band = np.asarray(band_2d, dtype=bool)
    seg_edge = np.asarray(seg_edge_2d, dtype=bool)

    if not band.any():
        empty2d = np.zeros_like(band, dtype=bool)
        return empty2d, empty2d, empty2d

    edt = distance_transform_edt(band)
    middle = band & (edt == maximum_filter(edt, size=3)) & (edt >= float(min_middle_half_width_px))

    edge = band & ~binary_erosion(band, iterations=1)
    if not edge.any():
        empty2d = np.zeros_like(band, dtype=bool)
        return middle, empty2d, empty2d

    if not seg_edge.any():
        empty2d = np.zeros_like(band, dtype=bool)
        return middle, empty2d, edge

    dist_to_seg = distance_transform_edt(~seg_edge)
    inner_edge = edge & (dist_to_seg <= 2.0)
    outer_edge = edge & ~inner_edge
    return middle, inner_edge, outer_edge


def get_band_middle_inner_outer_stack(
    band_stack: np.ndarray,
    seg_edge_stack: np.ndarray,
    min_middle_half_width_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply 2D band decomposition slice-by-slice on a stacked 3D array (z, y, x)."""
    band = np.asarray(band_stack, dtype=bool)
    seg_edge = np.asarray(seg_edge_stack, dtype=bool)

    middle = np.zeros_like(band, dtype=bool)
    inner_edge = np.zeros_like(band, dtype=bool)
    outer_edge = np.zeros_like(band, dtype=bool)
    for z in range(band.shape[0]):
        middle[z], inner_edge[z], outer_edge[z] = get_band_middle_inner_outer_2d(
            band[z],
            seg_edge[z],
            min_middle_half_width_px=min_middle_half_width_px,
        )
    return middle, inner_edge, outer_edge


def get_band_middle_inner_outer(
    band_mask: np.ndarray,
    seg_edge_mask: np.ndarray,
    min_middle_half_width_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible wrapper: 2D direct, otherwise slice-wise stack."""
    band = np.asarray(band_mask)
    if band.ndim == 2:
        return get_band_middle_inner_outer_2d(
            band_mask,
            seg_edge_mask,
            min_middle_half_width_px=min_middle_half_width_px,
        )
    return get_band_middle_inner_outer_stack(
        band_mask,
        seg_edge_mask,
        min_middle_half_width_px=min_middle_half_width_px,
    )


def _subsample_points(coords: np.ndarray, spacing: int) -> np.ndarray:
    """Greedy spatial subsampling: keep points at least `spacing` pixels apart."""
    if len(coords) == 0:
        return coords
    kept = [coords[0]]
    for pt in coords[1:]:
        if all(np.linalg.norm(pt - k) >= spacing for k in kept):
            kept.append(pt)
    return np.array(kept)


def compute_band_asymmetry_prompts_2d(
    pred_edge_2d: np.ndarray,
    pred_mask_2d: np.ndarray,
    binary_mask_2d: np.ndarray,
    asymmetry_threshold: float = 2.0,
    sample_spacing: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (positive_points, negative_points) as (N, 2) row/col arrays.

    At each segmentation edge pixel the outer band depth and inner band depth
    are compared.  Where the difference exceeds `asymmetry_threshold` pixels:
      - outer > inner  →  positive prompt  (model likely under-segments)
      - inner > outer  →  negative prompt  (model likely over-segments)
    """
    pred_edge = np.asarray(pred_edge_2d, dtype=bool)
    pred_mask = np.asarray(pred_mask_2d, dtype=bool)
    binary_mask = np.asarray(binary_mask_2d, dtype=bool)

    empty = np.empty((0, 2), dtype=np.float32)
    edge_coords = np.argwhere(pred_edge)
    if len(edge_coords) == 0 or not binary_mask.any():
        return empty, empty

    outer_band = binary_mask & ~pred_mask
    inner_band = binary_mask & pred_mask

    # Distance of each pixel from the prediction boundary, measured outward/inward.
    # edt_outside[p] = how far p sits outside pred_mask (0 inside pred_mask)
    # edt_inside[p]  = how far p sits inside  pred_mask (0 outside pred_mask)
    zeros = np.zeros(binary_mask.shape, dtype=float)
    edt_outside = distance_transform_edt(~pred_mask) if outer_band.any() else zeros
    edt_inside  = distance_transform_edt( pred_mask) if inner_band.any() else zeros

    # Depth maps: nonzero only within each band half.
    outer_depth_map = edt_outside * outer_band
    inner_depth_map = edt_inside  * inner_band

    # At each seg-edge pixel, take the max depth found in a local neighbourhood.
    # The neighbourhood must be large enough to span the band — use a generous window.
    nbr = max(11, int(asymmetry_threshold) * 4 + 1)
    local_outer = maximum_filter(outer_depth_map, size=nbr)
    local_inner = maximum_filter(inner_depth_map, size=nbr)

    ys, xs = edge_coords[:, 0], edge_coords[:, 1]
    asymmetry = local_outer[ys, xs] - local_inner[ys, xs]

    pos_pts = _subsample_points(edge_coords[asymmetry >  asymmetry_threshold], sample_spacing)
    neg_pts = _subsample_points(edge_coords[asymmetry < -asymmetry_threshold], sample_spacing)
    return pos_pts, neg_pts


def compute_band_asymmetry_prompts_stack(
    pred_edge_stack: np.ndarray,
    pred_mask_stack: np.ndarray,
    binary_mask_stack: np.ndarray,
    asymmetry_threshold: float = 2.0,
    sample_spacing: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply asymmetry prompt detection slice-by-slice. Returns (N, 3) arrays (z, y, x)."""
    all_pos, all_neg = [], []
    for z in range(pred_edge_stack.shape[0]):
        pos, neg = compute_band_asymmetry_prompts_2d(
            pred_edge_stack[z],
            pred_mask_stack[z],
            binary_mask_stack[z],
            asymmetry_threshold=asymmetry_threshold,
            sample_spacing=sample_spacing,
        )
        if len(pos):
            all_pos.append(np.column_stack([np.full(len(pos), z), pos]))
        if len(neg):
            all_neg.append(np.column_stack([np.full(len(neg), z), neg]))

    pos_all = np.concatenate(all_pos, axis=0) if all_pos else np.empty((0, 3), dtype=np.float32)
    neg_all = np.concatenate(all_neg, axis=0) if all_neg else np.empty((0, 3), dtype=np.float32)
    return pos_all, neg_all


def load_case_data(subject_index: int):
    """Load image, prediction, and uncertainty map for a subject."""
    project_root = Path.cwd()
    data_root = project_root / "data" / "LUNDPROBE"
    split = "ExtendedSamples"
    image_folder = "MR_StorT2"

    split_dir = data_root / split
    subjects = sorted([path.name for path in split_dir.iterdir() if path.is_dir()])

    subject_id = subjects[subject_index]
    case_dir = split_dir / subject_id / image_folder

    img_path = case_dir / "image.nii.gz"
    gt_mask_path = case_dir / "mask_CTVT_427.nii.gz"
    pred_mask_path = case_dir / "nnUNetOutput/mask_CTVT_427_nnUNet.nii.gz"
    uncertainty_map_path = case_dir / "nnUNetOutput/mask_CTVT_427_nnUNet_uncertaintyMap.nii.gz"

    # Load with SimpleITK and convert to numpy arrays
    img_itk = sitk.ReadImage(str(img_path))
    gt_mask_itk = sitk.ReadImage(str(gt_mask_path))
    pred_mask_itk = sitk.ReadImage(str(pred_mask_path))
    uncertainty_map_itk = sitk.ReadImage(str(uncertainty_map_path))

    # Convert to numpy arrays — shape is (z, y, x)
    img = sitk.GetArrayFromImage(img_itk).astype(np.float32)
    gt_mask = (sitk.GetArrayFromImage(gt_mask_itk) > 0).astype(np.uint8)
    pred_mask = (sitk.GetArrayFromImage(pred_mask_itk) > 0).astype(np.uint8)
    uncertainty_map = sitk.GetArrayFromImage(uncertainty_map_itk).astype(np.float32)

    return img, gt_mask, pred_mask, uncertainty_map, uncertainty_map_itk, case_dir


def main():
    parser = argparse.ArgumentParser(
        description="View MRI image, nnUNet segmentation, and uncertainty map in Napari."
    )
    parser.add_argument(
        "--subject_index",
        type=int,
        default=None,
        help="Subject index to load (integer).",
    )
    parser.add_argument(
        "--subject_id",
        type=str,
        default=None,
        help="Subject folder name to load (string).",
    )
    args = parser.parse_args()

    # Resolve which subject to load, or print the list and exit
    project_root = Path.cwd()
    split_dir = project_root / "data" / "LUNDPROBE" / "ExtendedSamples"
    subjects = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])

    if args.subject_index is None and args.subject_id is None:
        print("Available subjects (pass one of these):")
        for i, name in enumerate(subjects):
            print(f"  --subject_index {i:>3d}   --subject_id {name}")
        parser.exit(0)

    if args.subject_id is not None:
        if args.subject_id not in subjects:
            parser.error(f"Subject '{args.subject_id}' not found. Run without arguments to list all subjects.")
        subject_index = subjects.index(args.subject_id)
    else:
        subject_index = args.subject_index

    #Load data
    img, gt_mask, pred_mask, uncertainty_map, uncertainty_map_itk, case_dir = load_case_data(subject_index)

    #Mask out zero values in uncertainty map so they don't darken the display
    #This preserves the original data but makes 0 values transparent in the viewer
    uncertainty_masked = np.ma.masked_equal(uncertainty_map, 0.0)

    # Use absolute threshold bounds from 0 to the map maximum.
    unc_global_min = 0.0
    unc_global_max = float(uncertainty_map.max())
    if unc_global_max <= unc_global_min:
        unc_global_max = unc_global_min + 1e-6

    #Keep initial slider values strictly inside widget bounds to avoid
    #floating-point boundary errors during magicgui initialization.
    slider_eps = max(1e-6, (unc_global_max - unc_global_min) * 1e-6)

    #Initialize thresholds to full range [0, max].
    unc_min = unc_global_min
    unc_max = max(unc_global_min, unc_global_max - slider_eps)

    #Create Napari viewer
    viewer = napari.Viewer()

    #Add layers
    viewer.add_image(img, name="MRI image", colormap="gray")
    viewer.add_labels(gt_mask, name="Ground truth", opacity=0.5)
    viewer.add_labels(pred_mask, name="nnUNet segmentation", opacity=0.5)
    
    #Add uncertainty map with masked zeros and full-range contrast limits
    unc_layer = viewer.add_image(
        uncertainty_masked,
        name="Uncertainty map",
        colormap="magma",
        opacity=0.6,
        contrast_limits=[unc_global_min, unc_global_max],
    )

    # Pre-compute the prediction edge (static — doesn't depend on threshold)
    binary_threshold_default = min(0.1, unc_global_max)
    pred_edge_mask = build_prediction_edge_mask(pred_mask).astype(np.uint8)

    # Add layers with empty data — only populated when the user presses Run
    empty = np.zeros_like(uncertainty_map, dtype=np.uint8)
    binary_unc_layer = viewer.add_labels(
        empty,
        name="Binary uncertainty mask",
        opacity=0.5,
        visible=False,
    )
    pred_edge_layer = viewer.add_labels(
        empty,
        name="Prediction edge",
        opacity=0.9,
        visible=False,
    )
    band_middle_layer = viewer.add_labels(
        empty,
        name="Band middle (EDT ridge)",
        opacity=0.9,
        visible=False,
    )
    inner_band_edge_layer = viewer.add_labels(
        empty,
        name="Inner band edge",
        opacity=0.9,
        visible=False,
    )
    outer_band_edge_layer = viewer.add_labels(
        empty,
        name="Outer band edge",
        opacity=0.9,
        visible=False,
    )
    pos_prompt_layer = viewer.add_points(
        np.empty((0, 3)),
        name="Positive prompts",
        face_color="lime",
        size=1,
        visible=False,
    )
    neg_prompt_layer = viewer.add_points(
        np.empty((0, 3)),
        name="Negative prompts",
        face_color="red",
        size=1,
        visible=False,
    )

    @magicgui(
        binary_threshold={
            "widget_type": "FloatSlider",
            "value": binary_threshold_default,
            "min": unc_global_min,
            "max": unc_global_max,
            "step": slider_eps,
        }
    )
    def generate_binary_mask_and_edge(binary_threshold: float):
        """Generate binary uncertainty mask and show prediction edge."""
        binary_mask = (uncertainty_map >= binary_threshold).astype(np.uint8)
        middle, inner_edge, outer_edge = get_band_middle_inner_outer_stack(binary_mask, pred_edge_mask)

        binary_unc_layer.data = binary_mask
        pred_edge_layer.data = pred_edge_mask
        band_middle_layer.data = middle.astype(np.uint8)
        inner_band_edge_layer.data = inner_edge.astype(np.uint8)
        outer_band_edge_layer.data = outer_edge.astype(np.uint8)

        binary_unc_layer.visible = True
        pred_edge_layer.visible = True
        band_middle_layer.visible = True
        inner_band_edge_layer.visible = True
        outer_band_edge_layer.visible = True

        count = int(binary_mask.sum())
        print(
            f"Binary threshold: {binary_threshold:.4f} | "
            f"Binary voxels: {count} | "
            f"Edge voxels: {int(pred_edge_mask.sum())} | "
            f"Middle voxels: {int(middle.sum())} | "
            f"Inner edge voxels: {int(inner_edge.sum())} | "
            f"Outer edge voxels: {int(outer_edge.sum())}"
        )

    #Add the control widget to the viewer
    viewer.window.add_dock_widget(generate_binary_mask_and_edge, area="right", name="Generate mask")

    @magicgui(
        call_button="Generate prompts",
        asymmetry_threshold={
            "widget_type": "FloatSpinBox",
            "value": 2.0,
            "min": 0.0,
            "max": 50.0,
            "step": 0.5,
            "label": "Asymmetry threshold (px)",
        },
        sample_spacing={
            "widget_type": "SpinBox",
            "value": 10,
            "min": 1,
            "max": 100,
            "label": "Min spacing between prompts (px)",
        },
    )
    def generate_prompts(asymmetry_threshold: float, sample_spacing: int):
        """Compute positive/negative prompts from band asymmetry."""
        binary_mask = binary_unc_layer.data
        if not binary_mask.any():
            print("Generate the binary mask first.")
            return

        pos_pts, neg_pts = compute_band_asymmetry_prompts_stack(
            pred_edge_mask, pred_mask, binary_mask,
            asymmetry_threshold=asymmetry_threshold,
            sample_spacing=sample_spacing,
        )
        pos_prompt_layer.data = pos_pts
        neg_prompt_layer.data = neg_pts
        pos_prompt_layer.visible = True
        neg_prompt_layer.visible = True
        print(
            f"Asymmetry threshold: {asymmetry_threshold} px | "
            f"Spacing: {sample_spacing} px | "
            f"Pos prompts: {len(pos_pts)} | Neg prompts: {len(neg_pts)}"
        )

    viewer.window.add_dock_widget(generate_prompts, area="right", name="Generate prompts")

    @magicgui(call_button="Save binary mask")
    def save_binary_mask():
        """Save the current binary uncertainty mask as a NIfTI file."""
        binary_mask = binary_unc_layer.data
        if not binary_mask.any():
            print("No binary mask generated yet — press 'Run' first.")
            return
        threshold_val = generate_binary_mask_and_edge.binary_threshold.value
        out_path = case_dir / "nnUNetOutput" / f"binary_mask_thr{threshold_val:.4f}.nii.gz"
        out_itk = sitk.GetImageFromArray(binary_mask.astype(np.uint8))
        out_itk.CopyInformation(uncertainty_map_itk)
        sitk.WriteImage(out_itk, str(out_path))
        print(f"Saved → {out_path}")

    viewer.window.add_dock_widget(save_binary_mask, area="right", name="Save mask")

    @magicgui(call_button="Save prompts")
    def save_prompts():
        """Save positive/negative prompt points to an .npz file for use in Segmenter_App."""
        pos_pts = np.asarray(pos_prompt_layer.data, dtype=np.float32)
        neg_pts = np.asarray(neg_prompt_layer.data, dtype=np.float32)
        if len(pos_pts) == 0 and len(neg_pts) == 0:
            print("No prompts generated yet — press 'Generate prompts' first.")
            return
        out_path = case_dir / "nnUNetOutput" / "prompts.npz"
        np.savez(str(out_path), positive=pos_pts, negative=neg_pts)
        print(f"Saved → {out_path}  ({len(pos_pts)} positive, {len(neg_pts)} negative)")

    viewer.window.add_dock_widget(save_prompts, area="right", name="Save prompts")

    #Keep the viewer window open
    napari.run()


if __name__ == "__main__":
    main()
