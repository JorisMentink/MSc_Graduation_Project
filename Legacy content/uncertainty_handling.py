
import numpy as np
from scipy.ndimage import binary_erosion, center_of_mass
import matplotlib.pyplot as plt
from pathlib import Path

def extract_bands(seg, unc_map):
    """
    Function that extracts:
    - seg_edge: the edge of the segmentation
    - unc_inner: the inner part of the uncertainty band (uncertainty pixels that are inside the segmentation, excluding the edge)
    - unc_outer: the outer part of the uncertainty band (uncertainty pixels that are outside the segmentation)
    - unc_edge_inner: the inner edge of the uncertainty band (uncertainty pixels on the inner edge of the segmentation)
    - unc_edge_outer: the outer edge of the uncertainty band (uncertainty pixels on the outer edge of the segmentation)
    """

    #Change maps into boolean for easier operations
    seg = seg.astype(bool)
    unc_map = unc_map.astype(bool)

    seg_edge = seg & ~binary_erosion(seg) #Finds edge of the segmentation by isolating eroded pixels

    unc_inner = unc_map & seg & ~seg_edge #Finds all uncertainty pixels on the inside of the segmentation, excluding the edge
    unc_outer = unc_map & ~seg #Finds all uncertainty pixels on the outside of the segmentation

    unc_band_edges = binary_erosion(unc_map)
    
    unc_edge_inner = (unc_inner & ~binary_erosion(unc_inner)) & ~unc_band_edges #Inner edge of the uncertainty band
    unc_edge_outer = (unc_outer & ~binary_erosion(unc_outer)) & ~unc_band_edges #Outer edge of the uncertainty

    return seg_edge, unc_inner, unc_outer, unc_edge_inner, unc_edge_outer

def raycast_band_lengths_mm(
    seg,
    unc_map,
    unc_inner,
    seg_edge,
    unc_outer,
    pixel_spacing=(1.0, 1.0),
    angle_step=1,
    step_mm=None,
    pad=5,
    debug_dir=None,
    background_image=None,
    figsize=(6, 6),
):
    seg = seg.astype(bool)
    unc_map = unc_map.astype(bool)
    unc_inner = unc_inner.astype(bool)
    seg_edge = seg_edge.astype(bool)
    unc_outer = unc_outer.astype(bool)

    spacing_y = pixel_spacing[1]
    spacing_x = pixel_spacing[2]

    if not seg.any():
        raise ValueError("Segmentation is empty.")

    if not unc_map.any():
        raise ValueError("Uncertainty map is empty.")

    if step_mm is None:
        step_mm = min(spacing_y, spacing_x) / 4.0

    cy, cx = center_of_mass(seg)
    h, w = seg.shape

    ys, xs = np.where(unc_map)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    cy_box = (y_min + y_max) / 2
    cx_box = (x_min + x_max) / 2

    half_size = max(y_max - y_min, x_max - x_min) / 2
    half_size = int(np.ceil(half_size)) + pad

    y0 = max(0, int(np.floor(cy_box - half_size)))
    y1 = min(h, int(np.ceil(cy_box + half_size + 1)))
    x0 = max(0, int(np.floor(cx_box - half_size)))
    x1 = min(w, int(np.ceil(cx_box + half_size + 1)))

    # physical bbox limits in mm
    y0_mm = y0 * spacing_y
    y1_mm = (y1 - 1) * spacing_y
    x0_mm = x0 * spacing_x
    x1_mm = (x1 - 1) * spacing_x

    # center in mm
    cy_mm = cy * spacing_y
    cx_mm = cx * spacing_x

    # max possible ray length across image diagonal in mm
    max_length_mm = np.sqrt((h * spacing_y) ** 2 + (w * spacing_x) ** 2)

    angles = np.arange(0, 360, angle_step)

    inner_mm = []
    edge_mm = []
    outer_mm = []
    seg_radius_mm = []

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    for angle_deg in angles:
        theta = np.deg2rad(angle_deg)
        dy_mm = np.sin(theta)
        dx_mm = np.cos(theta)

        distances = np.arange(0, max_length_mm, step_mm)

        ray_y_mm = cy_mm + distances * dy_mm
        ray_x_mm = cx_mm + distances * dx_mm

        # stop at bbox in physical space
        valid = (
            (ray_y_mm >= y0_mm) & (ray_y_mm <= y1_mm) &
            (ray_x_mm >= x0_mm) & (ray_x_mm <= x1_mm)
        )

        ray_dist_mm = distances[valid]
        ray_y_mm = ray_y_mm[valid]
        ray_x_mm = ray_x_mm[valid]

        # convert back to image index coordinates
        ray_y = ray_y_mm / spacing_y
        ray_x = ray_x_mm / spacing_x

        # nearest-neighbor lookup
        iy = np.round(ray_y).astype(int)
        ix = np.round(ray_x).astype(int)

        inside = (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
        iy = iy[inside]
        ix = ix[inside]
        ray_y = ray_y[inside]
        ray_x = ray_x[inside]
        ray_dist_mm = ray_dist_mm[inside]

        seg_hits = seg[iy, ix]
        inner_hits = unc_inner[iy, ix]
        edge_hits = seg_edge[iy, ix]
        outer_hits = unc_outer[iy, ix]

        inner_len = np.sum(inner_hits) * step_mm
        edge_len = np.sum(edge_hits) * step_mm
        outer_len = np.sum(outer_hits) * step_mm

        if np.any(seg_hits):
            last_inside_idx = np.where(seg_hits)[0][-1]
            seg_radius = ray_dist_mm[last_inside_idx]
        else:
            seg_radius = np.nan

        inner_mm.append(inner_len)
        edge_mm.append(edge_len)
        outer_mm.append(outer_len)
        seg_radius_mm.append(seg_radius)

        if debug_dir is not None:
            fig, ax = plt.subplots(figsize=figsize)

            if background_image is not None:
                ax.imshow(background_image, cmap="gray")
            else:
                ax.imshow(seg.astype(float), cmap="gray")

            overlay_inner = np.zeros((h, w, 4), dtype=float)
            overlay_edge = np.zeros((h, w, 4), dtype=float)
            overlay_outer = np.zeros((h, w, 4), dtype=float)

            overlay_inner[unc_inner] = [0, 1, 0, 0.30]
            overlay_edge[seg_edge] = [1, 0, 0, 0.40]
            overlay_outer[unc_outer] = [0, 0, 1, 0.30]

            ax.imshow(overlay_inner)
            ax.imshow(overlay_edge)
            ax.imshow(overlay_outer)

            # bbox
            rect_x = [x0, x1 - 1, x1 - 1, x0, x0]
            rect_y = [y0, y0, y1 - 1, y1 - 1, y0]
            ax.plot(rect_x, rect_y, linewidth=1)

            # plotted as continuous line in index coordinates
            ax.plot(ray_x, ray_y, linewidth=2)
            ax.scatter([cx], [cy], marker="x", s=50)

            ax.set_title(
                f"{angle_deg:.1f}° | inner={inner_len:.2f} mm | "
                f"edge={edge_len:.2f} mm | outer={outer_len:.2f} mm | "
            )
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_aspect("equal")
            ax.axis("off")

            out_path = debug_dir / f"ray_{int(round(angle_deg)):03d}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    return {
        "center_of_mass_px": (cy, cx),
        "center_of_mass_mm": (cy_mm, cx_mm),
        "angles_deg": angles,
        "inner_mm": np.array(inner_mm),
        "edge_mm": np.array(edge_mm),
        "outer_mm": np.array(outer_mm),
        "seg_radius_mm": np.array(seg_radius_mm),
        "bbox_square_px": (y0, y1, x0, x1),
        "step_mm": step_mm,
    }


def process_rays(
    angles,
    inner_length,
    outer_length,
    edge_length,
    seg_radius,
    center_yx,
    spacing_yx,
    diff_thresh=2.0,
):
    cy, cx = center_yx
    sy, sx = spacing_yx

    mask = edge_length >= min(sy,sx) #Make sure every edge is at least one pixel wide 

    #Filter all arrays to only include valid rays
    angles_filtered = angles[mask]
    inner_filtered = inner_length[mask]
    outer_filtered = outer_length[mask]
    edge_filtered = edge_length[mask]
    seg_radius_filtered = seg_radius[mask]

    #Compute differential between outer and inner lengths
    edge_differential = outer_filtered - inner_filtered

    prompts_pos = []
    prompts_neg = []

    for i in range(len(angles_filtered)):
        diff = edge_differential[i]

        if np.isnan(seg_radius_filtered[i]):
            continue

        #Compute direction of ray in image coords
        theta = np.deg2rad(angles_filtered[i])
        dy = np.sin(theta)
        dx = np.cos(theta)

        #In case of diagnosed undersegmentation
        if diff >= diff_thresh:
            
            #Positive prompt generation
            r_mm_pos = seg_radius_filtered[i] + 0.5 * outer_filtered[i] # FAR AWAY PROMPTING. WAS 0.5
            y_pos = cy + (r_mm_pos * dy) / sy
            x_pos = cx + (r_mm_pos * dx) / sx
            prompts_pos.append((y_pos, x_pos))

            #TODO: NEGTIVE PROMPT COUNTERPART GENERATION? Kan wellicht maar hoeft niet
            # r_mm_neg = seg_radius_filtered[i] + 6 * outer_filtered[i] # NIETJES PRINCIPE NEGATIVE COUNTERPROMPT REALLY FAR AWAY
            # y_neg = cy + (r_mm_neg * dy) / sy
            # x_neg = cx + (r_mm_neg * dx) / sx
            # prompts_neg.append((y_neg, x_neg))

        #In case of diagnosed oversegmentation
        elif diff <= -diff_thresh:
            
            #Negative prompt generation
            r_mm_neg = seg_radius_filtered[i] - 0.25 * inner_filtered[i]
            y_neg = cy + (r_mm_neg * dy) / sy
            x_neg = cx + (r_mm_neg * dx) / sx
            prompts_neg.append((y_neg, x_neg))

            #Positive prompt counterpart generation (NIETJES PRINCIPE)
            r_mm_pos = seg_radius_filtered[i] - inner_filtered[i]
            y_pos = cy + (r_mm_pos * dy) / sy
            x_pos = cx + (r_mm_pos * dx) / sx
            prompts_pos.append((y_pos, x_pos))

    return {
        "positive_prompts": np.array(prompts_pos),
        "negative_prompts": np.array(prompts_neg),
    }


def determine_unc_thr(
    seg_3d,
    unc_3d,
    target_mm,
    spacing,
    step_fraction=0.02,
    angle_step=5,
    verbose=False,
):
    """
    Decrease threshold from max value until average band thickness >= target_mm
    """

    #Define maximum value for fractional downscaling
    max_val = float(np.max(unc_3d))
    step = step_fraction * max_val
    thr = max_val #Initially threshold from max value
    last_valid = None

    while thr > 0:
        band_values = []

        
        for z in range(seg_3d.shape[0]):
            #Skip empty slices
            seg = seg_3d[z]
            if not seg.any():
                continue
            unc_bin = unc_3d[z] >= thr
            if not unc_bin.any():
                continue

            #Extract band locations to determine thickness
            seg_edge, unc_inner, unc_outer, _, _ = extract_bands(seg, unc_bin)

            try:
                res = raycast_band_lengths_mm(
                    seg=seg,
                    unc_map=unc_bin,
                    unc_inner=unc_inner,
                    seg_edge=seg_edge,
                    unc_outer=unc_outer,
                    pixel_spacing=spacing,
                    angle_step=angle_step,
                    debug_dir=None,
                    step_mm=None,
                )

                #Determine band thickness average across rays and slices
                band = np.mean(res["inner_mm"] + res["outer_mm"])
                band_values.append(band)

            except ValueError:
                continue

        #Checks if no valid bands were found for this threshold across the 3D volume
        if len(band_values) == 0:
            thr -= step
            continue

        #Computes average band thickness across all valid slices for this threshold
        avg_band = np.mean(band_values)

        if verbose:
            print(f"thr={thr:.6f} | band={avg_band:.2f} mm")

        last_valid = (thr, avg_band)

        if avg_band >= target_mm:
            return thr, avg_band

        thr -= step #Lower threshold

    return last_valid



def generate_and_save_ray_prompts(
    pred_map_3d: np.ndarray,
    thr_unc_map: np.ndarray,
    img: np.ndarray,
    spacing,
    case_dir: Path,
    binary_threshold: float = 0.1,
    asymmetry_threshold: float = 2.0,
    edge_threshold: float = 0.4,
    angle_step: int = 5,
    step_mm: float = 0.2,
    make_debug_plots: bool = False,
    save_prompts_as_file = True,
    save_name: str | None = None,
):
    """
    Function that uses raycasting and band extraction function to generate positive and negative prompts to be used in MEDSAM2 segmentation pipeline.
    
    """
    
    pred_mask_stack = np.asarray(pred_map_3d).astype(bool)
    thresholded_unc = np.asarray(thr_unc_map)
    img_stack = np.asarray(img)

    if pred_mask_stack.shape != thresholded_unc.shape:
        raise ValueError("pred_mask_stack and thresholded_unc must have the same shape.")
    if pred_mask_stack.shape != img_stack.shape:
        raise ValueError("pred_mask_stack and img_stack must have the same shape.")

    all_pos = []
    all_neg = []

    for z in range(pred_mask_stack.shape[0]):
        seg = pred_mask_stack[z]
        unc = thresholded_unc[z]
        img = img_stack[z]

        if not seg.any() or not unc.any():
            continue

        seg_edge, unc_inner, unc_outer, _, _ = extract_bands(seg, unc)

        result = raycast_band_lengths_mm(
            seg=seg,
            unc_map=unc,
            unc_inner=unc_inner,
            seg_edge=seg_edge,
            unc_outer=unc_outer,
            pixel_spacing=spacing,
            angle_step=angle_step,
            step_mm=step_mm,
            debug_dir=(case_dir / "nnUNetOutput" / f"ray_debug_slice_{z:03d}") if make_debug_plots else None,
            background_image=img,
        )

        out = process_rays(
            angles=result["angles_deg"],
            inner_length=result["inner_mm"],
            outer_length=result["outer_mm"],
            edge_length=result["edge_mm"],
            seg_radius=result["seg_radius_mm"],
            center_yx=result["center_of_mass_px"],
            spacing_yx=(spacing[1], spacing[2]),
            diff_thresh=asymmetry_threshold,
        )

        pos = np.asarray(out["positive_prompts"], dtype=np.float32)
        neg = np.asarray(out["negative_prompts"], dtype=np.float32)

        if len(pos):
            all_pos.append(np.column_stack([np.full(len(pos), z, dtype=np.float32), pos]))
        if len(neg):
            all_neg.append(np.column_stack([np.full(len(neg), z, dtype=np.float32), neg]))

    pos_all = np.concatenate(all_pos, axis=0) if all_pos else np.empty((0, 3), dtype=np.float32)
    neg_all = np.concatenate(all_neg, axis=0) if all_neg else np.empty((0, 3), dtype=np.float32)

    out_dir = case_dir / "nnUNetOutput"
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_name is None:
        save_name = f"ray_prompts_thr{binary_threshold:.4f}_asym{asymmetry_threshold:.1f}.npz"

    out_path = out_dir / save_name

    if save_prompts_as_file:    
        np.savez_compressed(
            out_path,
            positive=pos_all.astype(np.float32),
            negative=neg_all.astype(np.float32),
            binary_threshold=np.float32(binary_threshold),
            asymmetry_threshold=np.float32(asymmetry_threshold),
            edge_threshold=np.float32(edge_threshold),
            angle_step=np.int32(angle_step),
            step_mm=np.float32(step_mm),
        )
        print(f"Saved prompts to: {out_path}")
        print(f"Positive prompts: {len(pos_all)}")
        print(f"Negative prompts: {len(neg_all)}")

    return pos_all, neg_all, out_path