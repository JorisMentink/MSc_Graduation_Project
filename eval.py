import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, center_of_mass

def get_surface(mask):
    return mask & ~binary_erosion(mask)


def surface_distances(pred, gt, spacing):
    pred_surf = get_surface(pred)
    gt_surf = get_surface(gt)

    dt_gt = distance_transform_edt(~gt_surf, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surf, sampling=spacing)

    d_pred_to_gt = dt_gt[pred_surf]
    d_gt_to_pred = dt_pred[gt_surf]

    return d_pred_to_gt, d_gt_to_pred, pred_surf, gt_surf


def compute_metrics(pred, gt, spacing=(1.0, 1.0, 1.0), surface_dice_tol=2.0, apl_tol=0.0):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    d_pred_to_gt, d_gt_to_pred, pred_surf, gt_surf = surface_distances(pred, gt, spacing)

    # HD95
    hd95 = np.percentile(np.concatenate([d_pred_to_gt, d_gt_to_pred]), 95)

    # MSD and ASSD
    msd = np.mean(d_pred_to_gt)
    assd = (np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred)) / 2

    # Surface Dice
    surface_dice = (
        np.sum(d_pred_to_gt <= surface_dice_tol) + np.sum(d_gt_to_pred <= surface_dice_tol)
    ) / (np.sum(pred_surf) + np.sum(gt_surf))

    # Centroid distance
    c_pred = np.array(center_of_mass(pred)) * np.array(spacing)
    c_gt = np.array(center_of_mass(gt)) * np.array(spacing)
    centroid_dist = np.linalg.norm(c_pred - c_gt)

    # Volume difference
    voxel_volume = np.prod(spacing)
    pred_vol = np.sum(pred) * voxel_volume
    gt_vol = np.sum(gt) * voxel_volume
    abs_vol_diff = abs(pred_vol - gt_vol)
    rel_vol_diff = 100 * (pred_vol - gt_vol) / gt_vol if gt_vol > 0 else np.nan

    # Very simple APL-like proxy
    apl_like = np.sum(d_gt_to_pred > apl_tol) * np.mean(spacing)

    return {
        "HD95_mm": hd95,
        "MSD_mm": msd,
        "ASSD_mm": assd,
        f"SurfaceDice@{surface_dice_tol}mm": surface_dice,
        "CentroidDistance_mm": centroid_dist,
        "AbsVolumeDifference_mm3": abs_vol_diff,
        "RelativeVolumeDifference_percent": rel_vol_diff,
        f"APL_like@{apl_tol}mm": apl_like,
    }