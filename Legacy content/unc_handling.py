import numpy as np
from scipy.ndimage import binary_erosion, center_of_mass
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk
from pathlib import Path
from uncertainty_util import determine_band_thickness_mm_normals_NEW, extract_bands, determine_band_thickness_mm_raycast, determine_band_thickness_mm_normals_subpixel, order_segmentation_pixels


class UG_prompter():

    def __init__(self,parentfolder,subject_nr=0,volume_of_interest="CTVT",verbose=False):

        self.parentfolder = Path(parentfolder)
        self.subject_nr = subject_nr
        self.volume_of_interest = volume_of_interest
        self.verbose = verbose

        if self.volume_of_interest not in ["CTVT","rectum"]:
            raise ValueError("Volume of interest must be either 'CTVT' or 'rectum'")

        self.subjects = sorted([p.name for p in self.parentfolder.iterdir() if p.is_dir()])
        self.subjectfolder = self.parentfolder / str(self.subjects[subject_nr]) / "MR_StorT2"
        
        #Load paths for image, mask and uncertainty map -following out-of-the-box LUNDPROBE formatting
        img_path = self.subjectfolder / "image.nii.gz"
        
        if self.volume_of_interest == "CTVT":
            mask_path = self.subjectfolder / "nnUNetOutput/mask_CTVT_427_nnUNet.nii.gz"
            unc_path = self.subjectfolder / "nnUNetOutput/mask_CTVT_427_nnUNet_uncertaintyMap.nii.gz"
        elif self.volume_of_interest == "rectum":
            mask_path = self.subjectfolder / "nnUNetOutput/mask_Rectum_nnUNet.nii.gz"
            unc_path = self.subjectfolder / "nnUNetOutput/mask_Rectum_nnUNet_uncertaintyMap.nii.gz"

        #Load data as arrays
        self.img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
        self.mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) > 0
        self.unc_map = sitk.GetArrayFromImage(sitk.ReadImage(str(unc_path)))

        #LOAD IMAGE SPACING
        img_itk = sitk.ReadImage(str(img_path))
        spacing_sitk = img_itk.GetSpacing()  # (x, y, z)
        self.img_spacing = spacing_sitk[::-1]  # (z, y, x)

    def threshold_uncertainty_map(self,unc_threshold=None,target_mm=3.0,step_fraction=0.05):

        #If threshold is manually set. use that
        if unc_threshold is not None:
            self.unc_map_bin  = self.unc_map >= unc_threshold
            return self.unc_map_bin
        
        else:
            max_val = float(np.max(self.unc_map))
            step = step_fraction * max_val
            thr = max_val #Initially threshold from max value
            last_valid = None

            while thr > 0:
                band_values = []

                for z in range(self.mask.shape[0]):
                    #Skipping empty slices
                    seg = self.mask[z]
                    unc_bin = self.unc_map[z] >= thr
                    if not seg.any() or not unc_bin.any():
                        continue
                    
                    seg_edge, unc_inner, unc_outer, _, __ = extract_bands(seg,unc_bin)

                    try:
                        res = determine_band_thickness_mm_raycast(
                            seg=seg,
                            unc_map=unc_bin,
                            unc_inner=unc_inner,
                            seg_edge=seg_edge,
                            unc_outer=unc_outer,
                            pixel_spacing=self.img_spacing,
                            angle_step=10,
                            step_mm=None,
                            pad=5
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

                if self.verbose:
                    print(f"thr={thr:.6f} | band={avg_band:.2f} mm")

                last_valid = (thr, avg_band)

                if avg_band >= target_mm:
                    self.thr = thr
                    self.unc_map_bin = self.unc_map >= self.thr
                    
                    return self.unc_map_bin
                
                thr -= step

            #Fallback if target_mm was never reached
            if last_valid is not None:
                self.thr, self.avg_band = last_valid
                if self.verbose:
                    print(
                        f"Target band thickness of {target_mm:.2f} mm was not reached. "
                        f"Using last valid threshold {self.thr:.6f} with band {self.avg_band:.2f} mm."
                    )
                self.unc_map_bin = self.unc_map >= self.thr
                return self.unc_map_bin

            raise ValueError("Could not determine a valid threshold.")
        
    def compute_band_thickness(self,method="raycast"):

        if method not in ["raycast","local_normal"]:
            raise ValueError("Method must be either 'raycast' or 'local_normal'")

        # Create list to store band thickness values for each slice
        self.band_thickness_per_slice = []
    
        for slice in range(self.unc_map_bin.shape[0]):
            
            seg = self.mask[slice]
            unc_bin = self.unc_map_bin[slice]
            if not seg.any() or not unc_bin.any():
                self.band_thickness_per_slice.append(0.0)
                continue

            seg_edge, unc_inner, unc_outer, _, __ = extract_bands(seg,unc_bin)

            try:
                if method == "raycast":
                    res = determine_band_thickness_mm_raycast(
                        seg=seg,
                        unc_map=unc_bin,
                        unc_inner=unc_inner,
                        seg_edge=seg_edge,
                        unc_outer=unc_outer,
                    pixel_spacing=self.img_spacing,
                    angle_step=10,
                    step_mm=None,
                    pad=5
                )
                
                elif method == "local_normal":
                    res = determine_band_thickness_mm_normals_subpixel(
                            seg=seg,
                            unc_inner=unc_inner,
                            seg_edge=seg_edge,
                            unc_outer=unc_outer,
                            pixel_spacing=self.img_spacing,
                            step_mm=0.1,
                            max_search_mm=20.0,
                            max_gap_mm=0.6,
                        )

                band = np.mean(res["inner_mm"] + res["outer_mm"])
                self.band_thickness_per_slice.append(band)

            except ValueError:
                self.band_thickness_per_slice.append(0.0)
                continue
        
        print(self.band_thickness_per_slice)

    def generate_prompts_boxes_and_points(self, band_threshold):
        """
        """
        prompts_by_slice = {}

        for z in range(self.mask.shape[0]):
            # Skip slices without a valid band thickness
            if z >= len(self.band_thickness_per_slice):
                continue

            if self.band_thickness_per_slice[z] <= band_threshold:
                continue

            seg = self.mask[z]
            unc = self.unc_map_bin[z].astype(bool)

            # Skip empty segmentation or empty uncertainty map
            if not seg.any() or not unc.any():
                continue

            # Positive point at segmentation center
            ys_seg, xs_seg = np.where(seg)
            cy = int(np.round(np.mean(ys_seg)))
            cx = int(np.round(np.mean(xs_seg)))

            positive_points = np.array([[cx, cy]], dtype=np.int32)  # (x, y)

            # Bounding box around full thresholded uncertainty map
            ys_unc, xs_unc = np.where(unc)
            y0 = int(np.min(ys_unc))
            y1 = int(np.max(ys_unc))
            x0 = int(np.min(xs_unc))
            x1 = int(np.max(xs_unc))

            boxes = np.array([[x0, y0, x1, y1]], dtype=np.int32)

            prompts_by_slice[z] = {
                "positive_points": positive_points,
                "boxes": boxes,
            }

        self.prompts_by_slice = prompts_by_slice
        return self.prompts_by_slice


    def generate_prompts_nietjes(
        self,
        unc_band_thr_mm=2.0,
        min_prompt_distance_px=10.0,
        max_prompts_per_slice=6,
        neighbor_offset=2,
        step_mm=0.1,
        max_search_mm=20.0,
    ):
        """
        """

        if not hasattr(self, "unc_map_bin"):
            raise AttributeError(
                "self.unc_map_bin does not exist yet. "
                "Call threshold_uncertainty_map(...) first."
            )

        prompts_by_slice = {}
        self.normals_by_slice = {}

        for z in range(self.mask.shape[0]):
            seg = self.mask[z].astype(bool)
            unc_bin = self.unc_map_bin[z].astype(bool)

            if not seg.any() or not unc_bin.any():
                continue

            seg_edge, unc_inner, unc_outer, _, _ = extract_bands(seg, unc_bin)

            if not seg_edge.any():
                continue

            try:
                result = determine_band_thickness_mm_normals_subpixel(
                    seg=seg,
                    unc_inner=unc_inner,
                    seg_edge=seg_edge,
                    unc_outer=unc_outer,
                    pixel_spacing=self.img_spacing,
                    neighbor_offset=neighbor_offset,
                    step_mm=step_mm,
                    max_search_mm=max_search_mm,
                )
            except ValueError:
                continue

            if result is None:
                continue

            midpoints_yx = result["midpoints"]
            normals_yx = result["normals"]
            inner_mm = result["inner_mm"]
            outer_mm = result["outer_mm"]
            total_mm = result["total_mm"]
            conf_core_mm = result["conf_core_mm"]

            valid = (
                np.isfinite(midpoints_yx).all(axis=1)
                & np.isfinite(normals_yx).all(axis=1)
                & np.isfinite(inner_mm)
                & np.isfinite(outer_mm)
                & np.isfinite(total_mm)
                & (total_mm >= unc_band_thr_mm)
            )

            if not np.any(valid):
                continue

            midpoints_yx = midpoints_yx[valid]
            normals_yx = normals_yx[valid]
            inner_mm = inner_mm[valid]
            outer_mm = outer_mm[valid]
            total_mm = total_mm[valid]
            conf_core_mm = conf_core_mm[valid]

            self.normals_by_slice[z] = {
                "midpoints": midpoints_yx,
                "normals": normals_yx,
            }

            candidates = []

            for i in range(len(midpoints_yx)):
                mid_yx = midpoints_yx[i]
                normal_yx = normals_yx[i]

                inner = inner_mm[i]
                outer = outer_mm[i]
                conf = conf_core_mm[i]

                if (inner + outer) < unc_band_thr_mm:
                    continue

                # Need a valid inner and outer band
                if inner <= 0 or outer <= 0:
                    continue

                # Positive prompt inward from the segmentation edge
                pos_yx = mid_yx - normal_yx * (inner + 0.25 * conf)

                # Negative prompt outward from the segmentation edge
                neg_yx = mid_yx + normal_yx * (2.0 * outer)

                for point_yx, label in [(pos_yx, 1), (neg_yx, 0)]:
                    y, x = point_yx

                    if (
                        y < 0
                        or y >= seg.shape[0]
                        or x < 0
                        or x >= seg.shape[1]
                    ):
                        continue

                    candidates.append(
                        {
                            "point_yx": point_yx,
                            "mid_yx": mid_yx,
                            "label": label,
                            "score": inner + outer,
                        }
                    )

            if len(candidates) == 0:
                continue

            candidates = sorted(
                candidates,
                key=lambda c: c["score"],
                reverse=True,
            )

            selected_points_yx = []
            selected_midpoints_yx = []
            selected_labels = []

            for candidate in candidates:
                point_yx = candidate["point_yx"]
                mid_yx = candidate["mid_yx"]

                if len(selected_midpoints_yx) > 0:
                    distances = [
                        np.linalg.norm(mid_yx - previous_mid_yx)
                        for previous_mid_yx in selected_midpoints_yx
                    ]

                    if min(distances) < min_prompt_distance_px:
                        continue

                selected_points_yx.append(point_yx)
                selected_midpoints_yx.append(mid_yx)
                selected_labels.append(candidate["label"])

                if len(selected_points_yx) >= max_prompts_per_slice:
                    break

            if len(selected_points_yx) == 0:
                continue

            selected_points_yx = np.asarray(
                selected_points_yx,
                dtype=np.float32,
            )
            selected_labels = np.asarray(
                selected_labels,
                dtype=np.int64,
            )

            # Convert from (y, x) to SAM/MedSAM2 format (x, y)
            selected_points_xy = selected_points_yx[:, ::-1]

            prompts_by_slice[z] = {
                "points": selected_points_xy,
                "point_labels": selected_labels,
                "bbox": None,
                "mask_input": None,
            }

            if self.verbose:
                n_pos = int(np.sum(selected_labels == 1))
                n_neg = int(np.sum(selected_labels == 0))
                print(
                    f"Slice {z}: {len(selected_labels)} prompts "
                    f"({n_pos} positive, {n_neg} negative)"
                )

        self.prompts_by_slice = prompts_by_slice
        return self.prompts_by_slice

    def generate_prompts_nietjes_2(
        self,
        unc_band_thr_mm=2.0,
        min_prompt_distance_px=10.0,
        max_prompts_per_slice=6,
        neighbor_offset=10,
        step_mm=0.1,
        max_search_mm=20.0,
    ):
        """
        Generate normal-based positive/negative prompts for all slices.

        Uses:
        - determine_band_thickness_mm_normals(...)
        - extract_bands(...)

        Stores:
        - self.prompts_by_slice
        - self.normals_by_slice

        Returns
        -------
        prompts_by_slice : dict
        """

        import numpy as np

        if not hasattr(self, "unc_map_bin"):
            raise AttributeError(
                "self.unc_map_bin does not exist yet. "
                "Call threshold_uncertainty_map(...) first."
            )

        prompts_by_slice = {}
        self.normals_by_slice = {}

        for z in range(self.mask.shape[0]):
            seg = self.mask[z].astype(bool)
            unc_bin = self.unc_map_bin[z].astype(bool)

            if not seg.any() or not unc_bin.any():
                continue

            seg_edge, unc_inner, unc_outer, _, _ = extract_bands(seg, unc_bin)

            if not seg_edge.any():
                continue

            try:
                result = determine_band_thickness_mm_normals_NEW(
                    seg=seg,
                    unc_inner=unc_inner,
                    seg_edge=seg_edge,
                    unc_outer=unc_outer,
                    ordered_edge_pixels=order_segmentation_pixels(seg_edge),
                    pixel_spacing=self.img_spacing,
                    # normal_window_mm=0.8,
                    # step_mm=0.2,
                    #max_search_mm=max_search_mm,
                    # max_gap_mm=0.6,
                )[0]
            except ValueError:
                continue

            if result is None:
                continue

            midpoints_yx = result["pixel_yx"]
            normals_yx = result["outer_normal_yx"]
            inner_mm = result["inner_mm"]
            outer_mm = result["outer_mm"]
            total_mm = result["total_mm"]
            #conf_core_mm = result["conf_core_mm"]

            valid = (
                np.isfinite(midpoints_yx).all(axis=1)
                & np.isfinite(normals_yx).all(axis=1)
                & np.isfinite(inner_mm)
                & np.isfinite(outer_mm)
                & np.isfinite(total_mm)
                #& np.isfinite(conf_core_mm)
                & (total_mm >= unc_band_thr_mm)
            )

            if not np.any(valid):
                continue

            midpoints_yx = midpoints_yx[valid]
            normals_yx = normals_yx[valid]
            inner_mm = inner_mm[valid]
            outer_mm = outer_mm[valid]
            total_mm = total_mm[valid]
            #conf_core_mm = conf_core_mm[valid]

            self.normals_by_slice[z] = {
                "midpoints": midpoints_yx,
                "normals": normals_yx,
                "inner_mm": inner_mm,
                "outer_mm": outer_mm,
                "total_mm": total_mm,
                #"conf_core_mm": conf_core_mm,
            }

            candidates = []

            for i in range(len(midpoints_yx)):
                mid_yx = midpoints_yx[i]
                normal_yx = normals_yx[i]

                inner = inner_mm[i]
                outer = outer_mm[i]
                total = total_mm[i]
                #conf = conf_core_mm[i]

                if total < unc_band_thr_mm:
                    continue

                # Positive prompt inward from the segmentation edge
                pos_yx = mid_yx - normal_yx * ((2.0 * inner)/self.img_spacing[1])

                # Negative prompt outward from the segmentation edge
                neg_yx = mid_yx + normal_yx * ((2.0 * outer)/self.img_spacing[1])

                for point_yx, label in [(pos_yx, 1), (neg_yx, 0)]:
                    y, x = point_yx

                    if (
                        y < 0
                        or y >= seg.shape[0]
                        or x < 0
                        or x >= seg.shape[1]
                    ):
                        continue

                    candidates.append(
                        {
                            "point_yx": point_yx,
                            "mid_yx": mid_yx,
                            "label": label,
                            "score": total,
                        }
                    )

            if len(candidates) == 0:
                continue

            candidates = sorted(
                candidates,
                key=lambda c: c["score"],
                reverse=True,
            )

            selected_points_yx = []
            selected_midpoints_yx = []
            selected_labels = []

            for candidate in candidates:
                point_yx = candidate["point_yx"]
                mid_yx = candidate["mid_yx"]

                if len(selected_midpoints_yx) > 0:
                    distances = [
                        np.linalg.norm(mid_yx - previous_mid_yx)
                        for previous_mid_yx in selected_midpoints_yx
                    ]

                    if min(distances) < min_prompt_distance_px:
                        continue

                selected_points_yx.append(point_yx)
                selected_midpoints_yx.append(mid_yx)
                selected_labels.append(candidate["label"])

                if len(selected_points_yx) >= max_prompts_per_slice:
                    break

            if len(selected_points_yx) == 0:
                continue

            selected_points_yx = np.asarray(selected_points_yx, dtype=np.float32)
            selected_labels = np.asarray(selected_labels, dtype=np.int64)

            # Convert from (y, x) to SAM/MedSAM2 format (x, y)
            selected_points_xy = selected_points_yx[:, ::-1]

            prompts_by_slice[z] = {
                "points": selected_points_xy,
                "point_labels": selected_labels,
                "bbox": None,
                "mask_input": None,
            }

            if self.verbose:
                n_pos = int(np.sum(selected_labels == 1))
                n_neg = int(np.sum(selected_labels == 0))
                print(
                    f"Slice {z}: {len(selected_labels)} prompts "
                    f"({n_pos} positive, {n_neg} negative)"
                )

        self.prompts_by_slice = prompts_by_slice
        return self.prompts_by_slice


    def plot_ordered_pixels(self, slice):
        import numpy as np
        import matplotlib.pyplot as plt

        seg = self.mask[slice]
        unc_bin = self.unc_map_bin[slice]

        seg_edge, unc_inner, unc_outer, _, __ = extract_bands(seg, unc_bin)

        ordered_pixels = order_segmentation_pixels(seg_edge)

        if ordered_pixels is None or len(ordered_pixels) == 0:
            print("No ordered pixels found.")
            return

        # Create order map
        order_map = np.full(seg_edge.shape, np.nan)

        for i, (y, x) in enumerate(ordered_pixels):
            order_map[y, x] = i

        # ---- CROP TO EDGE ----
        ys, xs = np.where(seg_edge)

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        # optional padding for visibility
        pad = 3
        y_min = max(0, y_min - pad)
        y_max = min(seg_edge.shape[0], y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(seg_edge.shape[1], x_max + pad)

        cropped = order_map[y_min:y_max, x_min:x_max]

        # ---- PLOT ----
        plt.figure(figsize=(6, 6))
        plt.imshow(cropped, cmap="viridis")
        plt.colorbar(label="Pixel order")
        plt.title(f"Ordered edge pixels (cropped), slice {slice}")
        plt.axis("equal")
        plt.axis("off")
        plt.show()

        print("Ordered pixels:", len(ordered_pixels))
        print("Total edge pixels:", np.sum(seg_edge))

