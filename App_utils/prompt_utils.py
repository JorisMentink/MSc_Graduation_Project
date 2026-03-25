import numpy as np
import SimpleITK as sitk

def get_points_for_slice(layer, slice_idx: int):
    """
    Napari points are stored as [z, y, x].
    Convert them to MedSAM format [x, y].
    """
    if len(layer.data) == 0:
        return np.empty((0, 2), dtype=np.float32)

    data = np.asarray(layer.data, dtype=np.float32)
    z = np.round(data[:, 0]).astype(int)
    slice_points = data[z == slice_idx]

    if len(slice_points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    return slice_points[:, [2, 1]].astype(np.float32)


def get_bbox_for_slice(layer, slice_idx: int) -> np.ndarray | None:
    """
    Napari rectangle vertices are stored as [z, y, x].
    Extract the latest rectangle drawn on the given slice and convert it to:
    [x_min, y_min, x_max, y_max]
    """
    if len(layer.data) == 0:
        return None

    matching = []
    for shape in layer.data:
        shape = np.asarray(shape, dtype=np.float32)
        if shape.ndim != 2 or shape.shape[1] != 3:
            continue

        z = np.round(shape[:, 0]).astype(int)
        if np.all(z == slice_idx):
            xs = shape[:, 2]
            ys = shape[:, 1]
            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
            matching.append(bbox)

    if len(matching) == 0:
        return None

    return matching[-1]


def get_mask_prompt_for_slice(layer, slice_idx: int) -> np.ndarray | None:
    """
    Return a binary 2D mask prompt for one slice, or None if empty.
    """
    mask2d = np.asarray(layer.data[slice_idx] > 0, dtype=np.uint8)
    if mask2d.sum() == 0:
        return None
    return mask2d


def load_mask_like_reference(mask_path: str, reference_shape: tuple[int, int, int]) -> np.ndarray:
    """
    Load a mask from disk and ensure it matches the image volume shape.
    """
    mask_itk = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(mask_itk)

    if mask.shape != reference_shape:
        raise ValueError(
            f"Loaded mask shape {mask.shape} does not match image shape {reference_shape}"
        )

    return (mask > 0).astype(np.uint8)


def generate_bbox_shapes_from_mask(mask_3d: np.ndarray, pad_px: int = 0, pad_frac: float = 0.0):
    """
    Create one Napari rectangle per slice where mask is present.

    Each rectangle is stored as 4 vertices in [z, y, x] format.
    Padding can be specified as:
      - pad_px: fixed number of pixels
      - pad_frac: fraction of the box width/height
    """
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask_3d.shape}")

    D, H, W = mask_3d.shape
    shapes = []

    for z in range(D):
        ys, xs = np.where(mask_3d[z] > 0)

        if len(xs) == 0 or len(ys) == 0:
            continue

        x_min = xs.min()
        x_max = xs.max()
        y_min = ys.min()
        y_max = ys.max()

        box_w = x_max - x_min + 1
        box_h = y_max - y_min + 1

        extra_x = int(round(box_w * pad_frac))
        extra_y = int(round(box_h * pad_frac))

        x_min = max(0, x_min - pad_px - extra_x)
        x_max = min(W - 1, x_max + pad_px + extra_x)
        y_min = max(0, y_min - pad_px - extra_y)
        y_max = min(H - 1, y_max + pad_px + extra_y)

        rect = np.array(
            [
                [z, y_min, x_min],
                [z, y_min, x_max],
                [z, y_max, x_max],
                [z, y_max, x_min],
            ],
            dtype=np.float32,
        )
        shapes.append(rect)

    return shapes


def collect_prompts_for_slice(slice_idx: int, pos_layer, neg_layer, box_layer, mask_prompt_layer):
    pos_pts = get_points_for_slice(pos_layer, slice_idx)
    neg_pts = get_points_for_slice(neg_layer, slice_idx)

    points_list = []
    labels_list = []

    if len(pos_pts) > 0:
        points_list.append(pos_pts)
        labels_list.append(np.ones(len(pos_pts), dtype=np.int64))

    if len(neg_pts) > 0:
        points_list.append(neg_pts)
        labels_list.append(np.zeros(len(neg_pts), dtype=np.int64))

    if len(points_list) > 0:
        points = np.concatenate(points_list, axis=0).astype(np.float32)
        point_labels = np.concatenate(labels_list, axis=0).astype(np.int64)
    else:
        points = None
        point_labels = None

    bbox = get_bbox_for_slice(box_layer, slice_idx)
    mask_input = get_mask_prompt_for_slice(mask_prompt_layer, slice_idx)

    return points, point_labels, bbox, mask_input


def get_all_prompted_slices(pos_layer, neg_layer, box_layer, mask_prompt_layer) -> list[int]:
    slices = set()

    if len(pos_layer.data) > 0:
        data = np.asarray(pos_layer.data, dtype=np.float32)
        slices.update(np.round(data[:, 0]).astype(int).tolist())

    if len(neg_layer.data) > 0:
        data = np.asarray(neg_layer.data, dtype=np.float32)
        slices.update(np.round(data[:, 0]).astype(int).tolist())

    if len(box_layer.data) > 0:
        for shape in box_layer.data:
            shape = np.asarray(shape, dtype=np.float32)
            if shape.ndim != 2 or shape.shape[1] != 3:
                continue
            z = np.round(shape[:, 0]).astype(int)
            if len(z) > 0 and np.all(z == z[0]):
                slices.add(int(z[0]))

    mask_data = np.asarray(mask_prompt_layer.data > 0, dtype=np.uint8)
    nonempty_slices = np.where(mask_data.reshape(mask_data.shape[0], -1).sum(axis=1) > 0)[0]
    slices.update(nonempty_slices.tolist())

    return sorted(slices)