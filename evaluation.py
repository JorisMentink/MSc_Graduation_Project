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

    def __init__(self, pred=None, gt=None, spacing=(1.0, 1.0, 1.0), segmentation: Segmentation = None):
        if segmentation is not None:
            self.pred = segmentation.predicted_seg.astype(bool)
            self.gt = segmentation.gt.astype(bool)
            self.spacing = tuple(segmentation.img_spacing)
        else:
            if pred is None or gt is None:
                raise ValueError("Provide either a segmentation or both pred and gt.")

            self.pred = pred.astype(bool)
            self.gt = gt.astype(bool)
            self.spacing = tuple(spacing)

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

        self._compute_surface_distances()

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
        Mean surface distance from prediction surface to ground truth surface.
        This is one-directional.
        """
        return np.mean(self.d_pred_to_gt)

    def assd(self):
        """
        Average symmetric surface distance.
        """
        return (np.mean(self.d_pred_to_gt) + np.mean(self.d_gt_to_pred)) / 2

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
            "HD_mm": self.hausdorff_distance(),
            "HD95_mm": self.hd95(),
            "MSD_mm": self.msd(),
            "ASSD_mm": self.assd(),
            f"SurfaceDice@{surface_dice_tol}mm": self.surface_dice(surface_dice_tol),
            "CentroidDistance_mm": self.centroid_distance(),
            "PredictionVolume_mm3": self.prediction_volume(),
            "GroundTruthVolume_mm3": self.ground_truth_volume(),
            "AbsVolumeDifference_mm3": self.absolute_volume_difference(),
            "RelativeVolumeDifference_percent": self.relative_volume_difference(),
        }


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