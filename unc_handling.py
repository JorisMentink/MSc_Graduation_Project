from unittest import result
from DataLoader import DataLoader
import numpy as np
from uncertainty_util import determine_band_thickness_mm_normals, determine_band_thickness_mm_normals, extract_bands, determine_band_thickness_mm_raycast, order_segmentation_pixels, check_prompt_validity


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

    def threshold_uncertainty_map(
        self,
        unc_threshold=None,
        target_mm=3.0,
        max_iter=20,
        tol_mm=0.005,
        method="raycast",
        mode="mean"
    ):

        # If threshold is manually set, use that
        if unc_threshold is not None:
            self.thr = unc_threshold
            self.unc_map_bin = self.unc_map >= self.thr
            return self.unc_map_bin

        max_val = float(np.max(self.unc_map))

        low = 0.0
        high = max_val

        best_thr = None
        best_band = None
        best_error = np.inf



        for i in range(max_iter):
            
            self.slice_value_records = []
            
            thr = (low + high) / 2.0
            band_values = []

            for z in range(self.mask.shape[0]):

                seg = self.mask[z]
                unc_bin = self.unc_map[z] >= thr

                if not seg.any() or not unc_bin.any():
                    continue

                seg_edge, unc_inner, unc_outer, _, __ = extract_bands(seg, unc_bin)

                #Determine band thickness 
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

                    values = res["inner_mm"] + res["outer_mm"]

                    band_values.extend(values)

                    self.slice_value_records.append(values)

                except ValueError:
                    continue


            self.all_records = band_values
            #Compute average band thickness across all slices
            if mode == "mean":
                avg_band = np.mean(band_values)
            elif mode == "median": #Can be better since edge-slices contain a lot of uncertainty and may skew results
                avg_band = np.median(band_values)
            else:
                raise ValueError("Mode must be either 'mean' or 'median'")
            
            error = abs(avg_band - target_mm)

            #Report itreation stats
            if self.verbose:
                print(
                    f"iter={i:02d} | thr={thr:.6f} | "
                    f"band={avg_band:.2f} mm | error={error:.2f}"
                )

            #Check if this is the best threshold so far
            if error < best_error:
                best_error = error
                best_thr = thr
                best_band = avg_band

            if error <= tol_mm:
                break

            if avg_band < target_mm:
                high = thr #Leads to decrease in threshold
            else:
                low = thr #Leads to increase in threshold

        if best_thr is None:
            raise ValueError("Could not determine a valid threshold.")

        self.thr = best_thr
        self.avg_band = best_band
        self.unc_map_bin = self.unc_map >= self.thr

        #Report selected threshold 
        if self.verbose:
            print(
                f"Selected threshold {self.thr:.6f} "
                f"with band thickness {self.avg_band:.2f} mm."
            )

        return self.unc_map_bin
        
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

                if res is None:
                    self.band_thickness_per_slice.append(0.0)
                    continue

                band = np.mean(res["inner_mm"] + res["outer_mm"])
                self.band_thickness_per_slice.append(band)

            except ValueError:
                self.band_thickness_per_slice.append(0.0)
                continue
        
        print(self.band_thickness_per_slice)

    def generate_prompts_boxes(self, band_threshold=0, pad=0):
        """
        Generate SAM2 prompts containing:
        - bounding box around segmentation + thresholded uncertainty region
        - dense mask prompt from self.mask
        """

        prompts_by_slice = {}

        for z in range(self.mask.shape[0]):

            if z >= len(self.band_thickness_per_slice):
                continue

            if self.band_thickness_per_slice[z] <= band_threshold:
                continue

            unc = self.unc_map_bin[z].astype(bool)
            mask_2d = self.mask[z].astype(bool)

            combined_region = unc | mask_2d

            if not combined_region.any():
                continue

            ys, xs = np.where(combined_region)

            H, W = combined_region.shape

            x0 = max(0, int(xs.min()) - pad)
            x1 = min(W - 1, int(xs.max()) + pad)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(H - 1, int(ys.max()) + pad)

            bbox = np.array([x0, y0, x1, y1], dtype=np.float32)

            prompts_by_slice[z] = {
                "points": None,
                "point_labels": None,
                "bbox": bbox,
                "mask_input": None,
            }

        self.prompts_by_slice = prompts_by_slice
        return prompts_by_slice


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

                if results is None:
                    continue

                print(
                        z,
                        "n_results:", len(results),
                        "max_total:", max(results["total_mm"]) if results else None,
                        "max_inner:", max(results["inner_mm"]) if results else None,
                        "max_outer:", max(results["outer_mm"]) if results else None,
                    )

                #STORE NORMALS FOR LATER PLOTTING/ANALYSIS
                self.normals_by_slice[z] = { "midpoints": np.asarray(results["pixel_yx"], dtype=float),
                    "inner_normals": np.asarray(results["inner_normal_yx"], dtype=float),
                    "outer_normals": np.asarray(results["outer_normal_yx"], dtype=float),}

                selected_points_yx = []
                selected_labels = []

                for pixel in range(len(results["pixel_index"])):
                    
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

                    # Positive prompt inward
                    pos_mm = mid_mm + inner_normal_yx * (inner_mm + 2)

                    # Negative prompt outward
                    neg_mm = mid_mm + outer_normal_yx * (outer_mm + 2)


                    pos_yx = np.array([
                        pos_mm[0] / self.img_spacing[1],
                        pos_mm[1] / self.img_spacing[2],
                    ])

                    neg_yx = np.array([
                        neg_mm[0] / self.img_spacing[1],
                        neg_mm[1] / self.img_spacing[2],
                    ])

                    #NEW PROMPT VALIDITY CHECKS.
                    if not check_prompt_validity(pos_yx, neg_yx, seg, unc_bin):
                       continue

                    selected_points_yx.extend([pos_yx, neg_yx])
                    selected_labels.extend([1, 0])

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
                    pos_mm = center_of_mass_mm + direction_yx * max(seg_mm[num] - (inner_mm[num]+2), 0.0)

                    # Negative prompt outside segmentation
                    neg_mm = center_of_mass_mm + direction_yx * (seg_mm[num] + (outer_mm[num]+2))

                    # Convert mm coordinates back to pixel coordinates: (y_mm, x_mm) -> (y_px, x_px)
                    pos_yx = np.array([
                        pos_mm[0] / self.img_spacing[1],
                        pos_mm[1] / self.img_spacing[2],
                    ])

                    neg_yx = np.array([
                        neg_mm[0] / self.img_spacing[1],
                        neg_mm[1] / self.img_spacing[2],
                    ])

                    if not check_prompt_validity(pos_yx, neg_yx, seg, unc_bin):
                        continue

                    selected_points_yx.extend([pos_yx, neg_yx])
                    selected_labels.extend([1, 0])

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
    
    def generate_prompts_grid(
        self,
        grid_size_pixels=20,
        negative_margin_pixels=30,
        band_threshold=None,
        max_positive_prompts_per_slice=None,
        max_negative_prompts_per_slice=None,
    ):
        """
        Generate grid prompts:
        - positive prompts: inside segmentation AND outside uncertainty map
        - negative prompts: outside segmentation AND outside uncertainty map

        grid_size_pixels controls the spacing between all grid points.
        """

        if not hasattr(self, "unc_map_bin"):
            raise AttributeError(
                "self.unc_map_bin does not exist yet. "
                "Call threshold_uncertainty_map(...) first."
            )

        prompts_by_slice = {}

        for z in range(self.mask.shape[0]):
            seg = self.mask[z].astype(bool)
            unc = self.unc_map_bin[z].astype(bool)

            if not seg.any():
                continue

            H, W = seg.shape

            combined = seg | unc
            ys, xs = np.where(combined)

            if len(ys) == 0:
                continue

            y0 = max(0, ys.min() - negative_margin_pixels)
            y1 = min(H, ys.max() + negative_margin_pixels + 1)
            x0 = max(0, xs.min() - negative_margin_pixels)
            x1 = min(W, xs.max() + negative_margin_pixels + 1)

            grid_y = np.arange(y0, y1, grid_size_pixels)
            grid_x = np.arange(x0, x1, grid_size_pixels)

            yy, xx = np.meshgrid(grid_y, grid_x, indexing="ij")

            yy = yy.ravel().astype(int)
            xx = xx.ravel().astype(int)

            positive_valid = seg[yy, xx] & (~unc[yy, xx])
            negative_valid = (~seg[yy, xx]) & (~unc[yy, xx])

            pos_yx = np.column_stack([yy[positive_valid], xx[positive_valid]])
            neg_yx = np.column_stack([yy[negative_valid], xx[negative_valid]])

            if max_positive_prompts_per_slice is not None:
                pos_yx = pos_yx[:max_positive_prompts_per_slice]

            if max_negative_prompts_per_slice is not None:
                neg_yx = neg_yx[:max_negative_prompts_per_slice]

            if len(pos_yx) == 0 and len(neg_yx) == 0:
                continue

            points_yx = np.vstack([pos_yx, neg_yx]).astype(np.float32)

            labels = np.concatenate([
                np.ones(len(pos_yx), dtype=np.int64),
                np.zeros(len(neg_yx), dtype=np.int64),
            ])

            # Convert from (y, x) to SAM format (x, y)
            points_xy = points_yx[:, ::-1]

            prompts_by_slice[z] = {
                "points": points_xy,
                "point_labels": labels,
                "bbox": None,
                "mask_input": None,
            }

        self.prompts_by_slice = prompts_by_slice
        return prompts_by_slice
