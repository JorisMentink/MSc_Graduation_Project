from unittest import result
from DataLoader import DataLoader
import numpy as np
from uncertainty_util import determine_band_thickness_mm_normals, determine_band_thickness_mm_normals, extract_bands, determine_band_thickness_mm_raycast, order_segmentation_pixels


class UG_prompter():

    def __init__(self, data : DataLoader):
        "Initializes UG_prompter with data from DataLoader instance. Call threshold_uncertainty_map(...) before generating prompts."

        if not isinstance(data, DataLoader):
            raise TypeError(
                "UG_prompter expects a DataLoader instance. "
                "Please create one first: dataloader = DataLoader(...)"
            )

        self.parentfolder = data.parentfolder
        self.subject_nr = data.subject_nr
        self.volume_of_interest = data.volume_of_interest
        self.verbose = data.verbose

        #Load data as arrays
        self.img = data.img
        self.mask = data.mask
        self.unc_map = data.unc_map
        self.gt = data.gt
        self.img_spacing = data.img_spacing

        if self.verbose:
            print(f"Initilialized UG_prompter for subject {self.subject_nr} with volume of interest '{self.volume_of_interest}'")
            print(f"Image shape: {self.img.shape}, Mask shape: {self.mask.shape}, Uncertainty map shape: {self.unc_map.shape}.")

    def threshold_uncertainty_map(self,unc_threshold=None,target_mm=3.0,step_fraction=0.05,mode="mean"):

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
                        if mode == "mean":
                            band = np.mean(res["inner_mm"] + res["outer_mm"])
                        elif mode == "median":
                            band = np.median(res["inner_mm"] + res["outer_mm"])
                        else:
                            raise ValueError("Mode must be either 'mean' or 'median'")
                        
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

        if method not in ["raycast","local_normals"]:
            raise ValueError("Method must be either 'raycast' or 'local_normals'")

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
                
                elif method == "local_normals":
                    res = determine_band_thickness_mm_normals(
                        seg=seg,
                        unc_inner=unc_inner,
                        seg_edge=seg_edge,
                        unc_outer=unc_outer,
                        ordered_edge_pixels=order_segmentation_pixels(seg_edge),
                        interpix_dist=2,
                        pixel_interval=1,
                        pixel_spacing=self.img_spacing)

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
        unc_band_thr_mm=2.0, #MINIMUM THICKNESS FOR PROMPT GENERATION
        min_prompt_distance_px=10.0,
        max_prompts_per_slice=1000,
        interpix_dist=2,
        pixel_interval=1,
        angle_step=10,
        method = "raycast",

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

            if method == "local_normals":
                try:
                    results = determine_band_thickness_mm_normals(
                            seg=seg,
                            unc_inner=unc_inner,
                            seg_edge=seg_edge,
                            unc_outer=unc_outer,
                            ordered_edge_pixels=order_segmentation_pixels(seg_edge),
                            interpix_dist=interpix_dist,
                            pixel_interval=pixel_interval,
                            pixel_spacing=self.img_spacing)
                except ValueError:
                    continue

                print(
                        z,
                        "n_results:", len(results),
                        "max_total:", max(results["total_mm"]) if results else None,
                        "max_inner:", max(results["inner_mm"]) if results else None,
                        "max_outer:", max(results["outer_mm"]) if results else None,
                    )
                

                if results is None:
                    continue

                #STORE NORMALS FOR LATER PLOTTING/ANALYSIS
                self.normals_by_slice[z] = { "midpoints": np.asarray(results["pixel_yx"], dtype=float),
                    "inner_normals": np.asarray(results["inner_normal_yx"], dtype=float),
                    "outer_normals": np.asarray(results["outer_normal_yx"], dtype=float),}

                selected_points_yx = []
                selected_labels = []

                for pixel in results["pixel_index"]:
                    mid_yx = results["pixel_yx"][pixel]

                    mid_mm = np.array([
                        mid_yx[0] * self.img_spacing[1],
                        mid_yx[1] * self.img_spacing[2],
                    ])

                    inner_normal_yx = results["inner_normal_yx"][pixel]
                    outer_normal_yx = results["outer_normal_yx"][pixel]

                    inner_mm = results["inner_mm"][pixel]
                    outer_mm = results["outer_mm"][pixel]
                    total_mm = results["total_mm"][pixel]

                    if total_mm < unc_band_thr_mm:
                        continue

                    if inner_mm <= 0 or outer_mm <= 0:
                        continue


                    #TODO: make this work with a scalar dependent on uncertainty. absolute values are not fixed.
                    # Positive prompt inward
                    pos_mm = mid_mm + inner_normal_yx * (6) #ABSOLUTE VALUE NOW. THIS SUCKS. LOL

                    # Negative prompt outward
                    neg_mm = mid_mm + outer_normal_yx * (12) # ABSOLUTE VALUE NOW. THIS SUCKS. LOL

                    pos_yx = np.array([
                        pos_mm[0] / self.img_spacing[1],
                        pos_mm[1] / self.img_spacing[2],
                    ])

                    neg_yx = np.array([
                        neg_mm[0] / self.img_spacing[1],
                        neg_mm[1] / self.img_spacing[2],
                    ])


                    for point_yx, label in [(pos_yx, 1), (neg_yx, 0)]:
                        y, x = point_yx

                        if y < 0 or y >= seg.shape[0] or x < 0 or x >= seg.shape[1]:
                            continue

                        selected_points_yx.append(point_yx)
                        selected_labels.append(label)

                    if len(selected_points_yx) >= max_prompts_per_slice:
                        break

                if len(selected_points_yx) == 0:
                    continue

                selected_points_yx = np.asarray(selected_points_yx, dtype=np.float32)
                selected_labels = np.asarray(selected_labels, dtype=np.int64)

                # Convert from (y, x) to SAM format (x, y)
                selected_points_xy = selected_points_yx[:, ::-1]

                prompts_by_slice[z] = {
                    "points": selected_points_xy,
                    "point_labels": selected_labels,
                    "bbox": None,
                    "mask_input": None,
                }

            elif method == "raycast":
                try:
                    results = determine_band_thickness_mm_raycast(
                        seg=seg,
                        unc_map=unc_bin,
                        unc_inner=unc_inner,
                        seg_edge=seg_edge,
                        unc_outer=unc_outer,
                        pixel_spacing=self.img_spacing,
                        angle_step=angle_step,
                        step_mm=None,
                        pad=5
                    )
                except ValueError:
                    continue
                
                selected_points_yx = []
                selected_labels = []

                center_of_mass_mm = np.asarray(results["center_of_mass_mm"], dtype=float)
                angles_deg = results["angles_deg"]
                
                seg_mm = results["seg_mm"]
                inner_mm = results["inner_mm"]
                edge_mm = results["edge_mm"]
                outer_mm = results["outer_mm"]
                band_total_mm = results["band_total_mm"]

#============================== SAVING RAYS FOR PLOTTING =======================================
                if not hasattr(self, "rays_by_slice"):
                    self.rays_by_slice = {}

                center_of_mass_px = np.asarray(results["center_of_mass_px"], dtype=float)

                ray_dirs_yx = np.column_stack([
                    np.sin(np.deg2rad(angles_deg)),
                    np.cos(np.deg2rad(angles_deg)),
                ])

                self.rays_by_slice[z] = {
                    "origin_yx": center_of_mass_px,
                    "directions_yx": ray_dirs_yx,
                    "angles_deg": angles_deg,
                    "seg_mm": seg_mm,
                    "inner_mm": inner_mm,
                    "edge_mm": edge_mm,
                    "outer_mm": outer_mm,
                    "band_total_mm": band_total_mm,
                }
#========================================================================================

                for num, angle in enumerate(angles_deg):

                    if band_total_mm[num] < unc_band_thr_mm:
                        continue

                    if seg_mm[num] <= 0:
                        continue

                    direction_yx = np.array([
                        np.sin(np.deg2rad(angle)),
                        np.cos(np.deg2rad(angle)),
                    ])

                    # Positive prompt inside segmentation
                    pos_mm = center_of_mass_mm + direction_yx * max(seg_mm[num] - 6.0, 0.0)

                    # Negative prompt outside segmentation
                    neg_mm = center_of_mass_mm + direction_yx * (seg_mm[num] + 12.0)

                    # Convert mm coordinates back to pixel coordinates: (y_mm, x_mm) -> (y_px, x_px)
                    pos_yx = np.array([
                        pos_mm[0] / self.img_spacing[1],
                        pos_mm[1] / self.img_spacing[2],
                    ])

                    neg_yx = np.array([
                        neg_mm[0] / self.img_spacing[1],
                        neg_mm[1] / self.img_spacing[2],
                    ])

                    for point_yx, label in [(pos_yx, 1), (neg_yx, 0)]:
                        y, x = point_yx

                        if y < 0 or y >= seg.shape[0] or x < 0 or x >= seg.shape[1]:
                            continue

                        selected_points_yx.append(point_yx)
                        selected_labels.append(label)

                    if len(selected_points_yx) >= max_prompts_per_slice:
                        break

                if len(selected_points_yx) == 0:
                    continue

                selected_points_yx = np.asarray(selected_points_yx, dtype=np.float32)
                selected_labels = np.asarray(selected_labels, dtype=np.int64)

                # Convert from (y, x) to SAM format (x, y)
                selected_points_xy = selected_points_yx[:, ::-1]

                prompts_by_slice[z] = {
                    "points": selected_points_xy,
                    "point_labels": selected_labels,
                    "bbox": None,
                    "mask_input": None,
                }

            else:
                raise ValueError(f"Method '{method}' not recognized. Use 'local_normals' or 'raycast'.")

        self.prompts_by_slice = prompts_by_slice
        return self.prompts_by_slice

            





