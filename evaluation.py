import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, center_of_mass

from pathlib import Path
import pandas as pd

from segmentation import Segmentation


class Evaluator:
    """
    Compute segmentation evaluation metrics between a prediction and ground truth.

    Can be initialized either with:
    - pred, gt, spacing directly
    - a Segmentation instance
    """

    def __init__(self, pred=None, gt=None, subject_name = "x", spacing=(1.0, 1.0, 1.0), segmentation: Segmentation = None):
        if segmentation is not None:
            self.pred = segmentation.predicted_seg.astype(bool)
            self.gt = segmentation.gt.astype(bool)
            self.spacing = tuple(segmentation.img_spacing)
            self.subject_name = segmentation.subject_name
        else:
            if pred is None or gt is None:
                raise ValueError("Provide either a segmentation or both pred and gt.")

            self.pred = pred.astype(bool)
            self.gt = gt.astype(bool)
            self.spacing = tuple(spacing)
            self.subject_name = subject_name

        if self.pred.shape != self.gt.shape:
            raise ValueError(
                f"Prediction and ground truth must have the same shape. "
                f"Got pred {self.pred.shape} and gt {self.gt.shape}."
            )

        if not self.pred.any():
            raise ValueError("Prediction mask is empty.")

        if not self.gt.any():
            raise ValueError("Ground truth mask is empty.")

        self.pred_surf = None
        self.gt_surf = None
        self.d_pred_to_gt = None
        self.d_gt_to_pred = None

        self.crop_to_foreground(margin=30)

        self._compute_surface_distances()

    def crop_to_foreground(self, margin=30):
        """
        Cropping function to remove irrelevant background. Speeds up surface distance compute immensely.
        """

        fg = self.pred | self.gt #Computes relevant pixels, as union of prediction and gt

        z, y, x = np.where(fg) #Finds indices of relevant pixels

        #Compute bounding box around relevant pixels. Adds margin cause perfect cropping stresses me out :)
        z0 = max(z.min() - margin, 0)
        z1 = min(z.max() + margin + 1, self.pred.shape[0])

        y0 = max(y.min() - margin, 0)
        y1 = min(y.max() + margin + 1, self.pred.shape[1])

        x0 = max(x.min() - margin, 0)
        x1 = min(x.max() + margin + 1, self.pred.shape[2])

        #Cropping
        self.pred = self.pred[z0:z1, y0:y1, x0:x1]
        self.gt   = self.gt[z0:z1, y0:y1, x0:x1]


    def get_surface(self, mask):
        return mask & ~binary_erosion(mask)

    def _compute_surface_distances(self):
        """
        Compute distances between prediction and ground truth surfaces,
        and store the surfaces and distances as attributes.
        """

        self.pred_surf = self.get_surface(self.pred)
        self.gt_surf = self.get_surface(self.gt)

        dt_gt = distance_transform_edt(~self.gt_surf, sampling=self.spacing)
        dt_pred = distance_transform_edt(~self.pred_surf, sampling=self.spacing)

        self.d_pred_to_gt = dt_gt[self.pred_surf]
        self.d_gt_to_pred = dt_pred[self.gt_surf]

    def hausdorff_distance(self):
        distances = np.concatenate([self.d_pred_to_gt, self.d_gt_to_pred])
        return np.max(distances)

    def hd95(self):
        distances = np.concatenate([self.d_pred_to_gt, self.d_gt_to_pred])
        return np.percentile(distances, 95)

    def msd(self):
        """
        Mean surface distance from prediction surface to ground truth surface.k
        This is one-directional.
        """
        return np.mean(self.d_pred_to_gt)

    def assd(self):
        """
        Average symmetric surface distance.
        """
        return (np.mean(self.d_pred_to_gt) + np.mean(self.d_gt_to_pred)) / 2

    def dice(self):
        intersection = np.sum(self.pred & self.gt)
        denominator = np.sum(self.pred) + np.sum(self.gt)

        return 2 * intersection / denominator

    def surface_dice(self, tolerance_mm=1.0):
        """
        Compute Surface Dice at a given tolerance.

        Counts how many prediction and ground truth surface points are within
        tolerance_mm of the opposite surface, divided by the total number of
        surface points.
        """

        numerator = (
            np.sum(self.d_pred_to_gt <= tolerance_mm)
            + np.sum(self.d_gt_to_pred <= tolerance_mm)
        )

        denominator = np.sum(self.pred_surf) + np.sum(self.gt_surf)

        return numerator / denominator

    def centroid_distance(self):
        c_pred = np.array(center_of_mass(self.pred)) * np.array(self.spacing)
        c_gt = np.array(center_of_mass(self.gt)) * np.array(self.spacing)

        return np.linalg.norm(c_pred - c_gt)

    def prediction_volume(self):
        voxel_volume = np.prod(self.spacing)
        return np.sum(self.pred) * voxel_volume

    def ground_truth_volume(self):
        voxel_volume = np.prod(self.spacing)
        return np.sum(self.gt) * voxel_volume

    def absolute_volume_difference(self):
        return abs(self.prediction_volume() - self.ground_truth_volume())

    def relative_volume_difference(self):
        gt_vol = self.ground_truth_volume()

        if gt_vol == 0:
            return np.nan

        return 100 * (self.prediction_volume() - gt_vol) / gt_vol

    def compute_all(self, surface_dice_tol=1.0):
        return {
            "subject name": self.subject_name,
            "HD_mm": self.hausdorff_distance(),
            "HD95_mm": self.hd95(),
            "MSD_mm": self.msd(),
            "ASSD_mm": self.assd(),
            "Dice": self.dice(),
            f"SurfaceDice@{surface_dice_tol}mm": self.surface_dice(surface_dice_tol),
            "CentroidDistance_mm": self.centroid_distance(),
            "PredictionVolume_mm3": self.prediction_volume(),
            "GroundTruthVolume_mm3": self.ground_truth_volume(),
            "AbsVolumeDifference_mm3": self.absolute_volume_difference(),
            "RelativeVolumeDifference_percent": self.relative_volume_difference(),
        }

class SliceEvaluator:
    """
    Compute 2D slice-wise segmentation metrics between prediction and ground truth.
    """

    def __init__(self, pred_slice, gt_slice, subject_name="x", slice_idx=None, spacing=(1.0, 1.0)):
        self.pred = pred_slice.astype(bool)
        self.gt = gt_slice.astype(bool)
        self.subject_name = subject_name
        self.slice_idx = slice_idx
        self.spacing = tuple(spacing)  # should be (y_spacing, x_spacing)

        if self.pred.shape != self.gt.shape:
            raise ValueError(
                f"Prediction and ground truth slice must have the same shape. "
                f"Got pred {self.pred.shape} and gt {self.gt.shape}."
            )

        if not self.pred.any():
            raise ValueError("Prediction mask is empty.")

        if not self.gt.any():
            raise ValueError("Ground truth mask is empty.")

        self.pred_surf = None
        self.gt_surf = None
        self.d_pred_to_gt = None
        self.d_gt_to_pred = None

        self.crop_to_foreground(margin=30)
        self._compute_surface_distances()

    def crop_to_foreground(self, margin=30):
        """
        Cropping function to remove irrelevant background.
        """

        fg = self.pred | self.gt

        y, x = np.where(fg)

        y0 = max(y.min() - margin, 0)
        y1 = min(y.max() + margin + 1, self.pred.shape[0])

        x0 = max(x.min() - margin, 0)
        x1 = min(x.max() + margin + 1, self.pred.shape[1])

        self.pred = self.pred[y0:y1, x0:x1]
        self.gt = self.gt[y0:y1, x0:x1]

    def get_surface(self, mask):
        return mask & ~binary_erosion(mask)

    def _compute_surface_distances(self):
        self.pred_surf = self.get_surface(self.pred)
        self.gt_surf = self.get_surface(self.gt)

        dt_gt = distance_transform_edt(~self.gt_surf, sampling=self.spacing)
        dt_pred = distance_transform_edt(~self.pred_surf, sampling=self.spacing)

        self.d_pred_to_gt = dt_gt[self.pred_surf]
        self.d_gt_to_pred = dt_pred[self.gt_surf]

    def hausdorff_distance(self):
        distances = np.concatenate([self.d_pred_to_gt, self.d_gt_to_pred])
        return np.max(distances)

    def hd95(self):
        distances = np.concatenate([self.d_pred_to_gt, self.d_gt_to_pred])
        return np.percentile(distances, 95)

    def msd(self):
        return np.mean(self.d_pred_to_gt)

    def assd(self):
        return (np.mean(self.d_pred_to_gt) + np.mean(self.d_gt_to_pred)) / 2

    def dice(self):
        intersection = np.sum(self.pred & self.gt)
        denominator = np.sum(self.pred) + np.sum(self.gt)

        return 2 * intersection / denominator

    def surface_dice(self, tolerance_mm=1.0):
        numerator = (
            np.sum(self.d_pred_to_gt <= tolerance_mm)
            + np.sum(self.d_gt_to_pred <= tolerance_mm)
        )

        denominator = np.sum(self.pred_surf) + np.sum(self.gt_surf)

        return numerator / denominator

    def centroid_distance(self):
        c_pred = np.array(center_of_mass(self.pred)) * np.array(self.spacing)
        c_gt = np.array(center_of_mass(self.gt)) * np.array(self.spacing)

        return np.linalg.norm(c_pred - c_gt)

    def prediction_area(self):
        pixel_area = np.prod(self.spacing)
        return np.sum(self.pred) * pixel_area

    def ground_truth_area(self):
        pixel_area = np.prod(self.spacing)
        return np.sum(self.gt) * pixel_area

    def absolute_area_difference(self):
        return abs(self.prediction_area() - self.ground_truth_area())

    def relative_area_difference(self):
        gt_area = self.ground_truth_area()

        if gt_area == 0:
            return np.nan

        return 100 * (self.prediction_area() - gt_area) / gt_area

    def compute_all(self, surface_dice_tol=1.0):
        return {
            "subject": self.subject_name,
            "slice_idx": self.slice_idx,
            "HD_mm": self.hausdorff_distance(),
            "HD95_mm": self.hd95(),
            "MSD_mm": self.msd(),
            "ASSD_mm": self.assd(),
            "Dice" : self.dice(),
            f"SurfaceDice@{surface_dice_tol}mm": self.surface_dice(surface_dice_tol),
            "CentroidDistance_mm": self.centroid_distance(),
            "PredictionArea_mm2": self.prediction_area(),
            "GroundTruthArea_mm2": self.ground_truth_area(),
            "AbsAreaDifference_mm2": self.absolute_area_difference(),
            "RelativeAreaDifference_percent": self.relative_area_difference(),
        }


def evaluate_slice_by_slice(
    pred,
    gt,
    uncertainty=None,
    spacing=(1.0, 1.0, 1.0),
    subject_name="x",
    surface_dice_tol=1.0,
):
    """
    Evaluate segmentation slice by slice.

    Assumes volume shape is (z, y, x).
    Uses in-plane spacing only: spacing[1], spacing[2].

    Only evaluates slices from the minimum to maximum slice index
    where the prediction exists.

    If uncertainty is provided, it should have the same shape as pred and gt.
    """

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.shape != gt.shape:
        raise ValueError(
            f"Prediction and ground truth must have the same shape. "
            f"Got pred {pred.shape} and gt {gt.shape}."
        )

    if uncertainty is not None:
        uncertainty = uncertainty.astype(bool)

        if uncertainty.shape != pred.shape:
            raise ValueError(
                f"Uncertainty map must have the same shape as prediction. "
                f"Got uncertainty {uncertainty.shape} and pred {pred.shape}."
            )

    # Find slice range where prediction exists
    pred_slice_has_pixels = np.any(pred, axis=(1, 2))
    valid_pred_slices = np.where(pred_slice_has_pixels)[0]

    if len(valid_pred_slices) == 0:
        raise ValueError("Prediction volume is empty. No slices can be evaluated.")

    min_slice_idx = int(valid_pred_slices.min())
    max_slice_idx = int(valid_pred_slices.max())

    slice_indices = range(min_slice_idx, max_slice_idx + 1)
    num_valid_slices = max_slice_idx - min_slice_idx + 1

    in_plane_spacing = (spacing[1], spacing[2])

    rows = []

    print(f"Evaluating slices {min_slice_idx} to {max_slice_idx}...")
    print(f"Number of evaluated slices: {num_valid_slices}")

    for local_idx, z in enumerate(slice_indices):
        pred_slice = pred[z]
        gt_slice = gt[z]

        if uncertainty is not None:
            unc_slice = uncertainty[z]
        else:
            unc_slice = np.zeros_like(pred_slice, dtype=bool)

        pred_pixels = np.sum(pred_slice)
        gt_pixels = np.sum(gt_slice)
        unc_pixels = np.sum(unc_slice)

        has_prediction = pred_pixels > 0
        has_ground_truth = gt_pixels > 0
        has_uncertainty = unc_pixels > 0

        prediction_without_gt = has_prediction and not has_ground_truth
        gt_without_prediction = has_ground_truth and not has_prediction
        both_empty = not has_prediction and not has_ground_truth
        both_present = has_prediction and has_ground_truth

        oversegmentation = pred_pixels > gt_pixels
        undersegmentation = pred_pixels < gt_pixels

        if both_present:
            try:
                evaluator = SliceEvaluator(
                    pred_slice=pred_slice,
                    gt_slice=gt_slice,
                    spacing=in_plane_spacing,
                    subject_name=subject_name,
                    slice_idx=z,
                )

                metrics = evaluator.compute_all(surface_dice_tol)
                error = np.nan

            except ValueError as e:
                metrics = {
                    "subject": subject_name,
                    "slice_idx": z,
                    "HD_mm": np.nan,
                    "HD95_mm": np.nan,
                    "MSD_mm": np.nan,
                    "ASSD_mm": np.nan,
                    f"SurfaceDice@{surface_dice_tol}mm": np.nan,
                    "CentroidDistance_mm": np.nan,
                }

                error = str(e)

        else:
            metrics = {
                "subject": subject_name,
                "slice_idx": z,
                "HD_mm": np.nan,
                "HD95_mm": np.nan,
                "MSD_mm": np.nan,
                "ASSD_mm": np.nan,
                f"SurfaceDice@{surface_dice_tol}mm": np.nan,
                "CentroidDistance_mm": np.nan,
            }

            if prediction_without_gt:
                error = "Prediction exists, but ground truth is empty."
            elif gt_without_prediction:
                error = "Ground truth exists, but prediction is empty."
            else:
                error = "Both prediction and ground truth are empty."

        metrics["PredictionPixels"] = pred_pixels
        metrics["GroundTruthPixels"] = gt_pixels
        metrics["UncertaintyPixels"] = unc_pixels

        metrics["PredictionToGroundTruthPixelRatio"] = (
            pred_pixels / gt_pixels if gt_pixels > 0 else np.nan
        )

        metrics["UncertaintyToPredictionPixelRatio"] = (
            unc_pixels / pred_pixels if pred_pixels > 0 else np.nan
        )

        metrics["UncertaintyToGroundTruthPixelRatio"] = (
            unc_pixels / gt_pixels if gt_pixels > 0 else np.nan
        )

        metrics["HasPrediction"] = has_prediction
        metrics["HasGroundTruth"] = has_ground_truth
        metrics["HasUncertainty"] = has_uncertainty
        metrics["PredictionWithoutGroundTruth"] = prediction_without_gt
        metrics["GroundTruthWithoutPrediction"] = gt_without_prediction
        metrics["BothEmpty"] = both_empty
        metrics["BothPresent"] = both_present
        metrics["Oversegmentation"] = oversegmentation
        metrics["Undersegmentation"] = undersegmentation
        metrics["error"] = error

        # Relative position within evaluated slice range
        metrics["relative slice idx"] = (
            local_idx / (num_valid_slices - 1)
            if num_valid_slices > 1
            else 0.0
        )

        rows.append(metrics)

    return pd.DataFrame(rows)

def compare_to_recontours(
    pred_seg,
    recontours,
    spacing=(1.0, 1.0, 1.0),
    subject_name="x",
    observer_names=None,
    surface_dice_tol=1.0,
):

    if observer_names is None:
        observer_names = [f"Observer_{i+1}" for i in range(len(recontours))]

    pred_seg = pred_seg.astype(bool)
    recontours = [r.astype(bool) for r in recontours]

    rows = []

    def evaluate(seg_a, seg_b, name_a, name_b, group):

        evaluator = Evaluator(
            pred=seg_a,
            gt=seg_b,
            spacing=spacing,
            subject_name=subject_name,
        )

        metrics = evaluator.compute_all(surface_dice_tol)

        metrics["Group"] = group
        metrics["Comparison"] = f"{name_a} vs {name_b}"

        return metrics

    # Clinician vs clinician
    for i in range(len(recontours)):
        for j in range(i + 1, len(recontours)):
            rows.append(
                evaluate(
                    recontours[i],
                    recontours[j],
                    observer_names[i],
                    observer_names[j],
                    "Clinician vs clinician",
                )
            )

    # Prediction vs clinician
    for obs_seg, obs_name in zip(recontours, observer_names):
        rows.append(
            evaluate(
                pred_seg,
                obs_seg,
                "Prediction",
                obs_name,
                "Prediction vs clinician",
            )
        )

    # Consensus
    vote_map = np.sum(np.stack(recontours, axis=0), axis=0)
    majority_threshold = int(np.ceil(len(recontours) / 2))
    consensus = vote_map >= majority_threshold

    # Clinician vs consensus
    for obs_seg, obs_name in zip(recontours, observer_names):
        rows.append(
            evaluate(
                obs_seg,
                consensus,
                obs_name,
                "Consensus",
                "Clinician vs consensus",
            )
        )

    # Prediction vs consensus
    rows.append(
        evaluate(
            pred_seg,
            consensus,
            "Prediction",
            "Consensus",
            "Prediction vs consensus",
        )
    )

    results_df = pd.DataFrame(rows)

    cols = [
        "Group",
        "Comparison",
        f"SurfaceDice@{surface_dice_tol}mm",
        "ASSD_mm",
        "HD95_mm",
        "CentroidDistance_mm",
        "RelativeVolumeDifference_percent",
    ]

    results_df = results_df[cols]

    return results_df, consensus, vote_map


# NOT PART OF THE CLASS. UTILITY FUNCTION TO SAVE RESULTS TO CSV/EXCEL
def save_evaluation_results(
    results_list,
    output_folder,
    filename="segmentation_evaluation_results",
):
    """
    Save a list of SegmentationEvaluator.compute_all() outputs to CSV and Excel.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results_list)

    csv_path = output_folder / f"{filename}.csv"
    excel_path = output_folder / f"{filename}.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)

    print(f"Saved CSV to:   {csv_path}")
    print(f"Saved Excel to: {excel_path}")

    return df