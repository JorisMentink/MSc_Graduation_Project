import os
from scipy import ndimage
import numpy as np
import SimpleITK as sitk

def keep_largest_3d_connected_component(mask_3d: np.ndarray):
    """
    Keep only the largest 3D connected foreground component.
    Input and output are binary masks of shape (D, H, W).
    """


    binary = mask_3d > 0
    labeled, num = ndimage.label(binary)

    if num == 0:
        return np.zeros_like(mask_3d, dtype=np.uint8)

    component_sizes = ndimage.sum(binary, labeled, index=np.arange(1, num + 1))
    largest_label = int(np.argmax(component_sizes)) + 1

    cleaned = (labeled == largest_label).astype(np.uint8)
    return cleaned

def keep_largest_3d_component_touching_prompted_slices(
    mask_3d: np.ndarray,
    prompted_slices: list[int],
) -> np.ndarray:
    """
    Keep the largest 3D connected foreground component that has at least
    one voxel on at least one prompted slice.

    If no component touches a prompted slice, return an empty mask.
    """
    from scipy import ndimage

    binary = mask_3d > 0
    labeled, num = ndimage.label(binary)

    if num == 0:
        return np.zeros_like(mask_3d, dtype=np.uint8)

    valid_labels = []

    for label_id in range(1, num + 1):
        component_mask = (labeled == label_id)
        if any(component_mask[z].any() for z in prompted_slices):
            valid_labels.append(label_id)

    if len(valid_labels) == 0:
        return np.zeros_like(mask_3d, dtype=np.uint8)

    largest_valid_label = max(
        valid_labels,
        key=lambda label_id: np.count_nonzero(labeled == label_id)
    )

    return (labeled == largest_valid_label).astype(np.uint8)


def save_mask_like_reference(mask_3d: np.ndarray, reference_itk, output_mask: str):
    mask_itk = sitk.GetImageFromArray(mask_3d.astype(np.uint8))
    mask_itk.CopyInformation(reference_itk)

    out_dir = os.path.dirname(os.path.abspath(output_mask))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sitk.WriteImage(mask_itk, output_mask)
    print("Saved mask to:", output_mask)