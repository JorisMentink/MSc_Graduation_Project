import os
import argparse
from contextlib import nullcontext

import numpy as np
import SimpleITK as sitk
from PIL import Image
import torch
import napari
from magicgui import magicgui
from magicgui.widgets import PushButton, Container
from napari.utils.notifications import show_info, show_error

from sam2.build_sam import build_sam2_video_predictor_npz


def normalize_mri_to_uint8(volume: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    vol = volume.astype(np.float32)
    lo, hi = np.percentile(vol, (p_low, p_high))
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + 1e-8)
    vol = (vol * 255.0).astype(np.uint8)
    return vol


def resize_grayscale_stack_to_rgb(volume_uint8: np.ndarray, image_size: int) -> np.ndarray:
    d, h, w = volume_uint8.shape
    out = np.zeros((d, 3, image_size, image_size), dtype=np.float32)
    for i in range(d):
        img_pil = Image.fromarray(volume_uint8[i])
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        arr = np.asarray(img_resized, dtype=np.float32) / 255.0
        out[i] = arr.transpose(2, 0, 1)
    return out


def keep_largest_3d_connected_component(mask_3d: np.ndarray):
    """
    Keep only the largest 3D connected foreground component.
    Input and output are binary masks of shape (D, H, W).
    """
    from scipy import ndimage

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


def generate_bbox_shapes_from_mask(mask_3d: np.ndarray, pad_px: int = 5, pad_frac: float = 0.0):
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


def run_medsam2_inference_from_arrays(
    vol: np.ndarray,
    predictor,
    image_size: int,
    prompts_by_slice: dict[int, tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]],
    p_low: float = 1.0,
    p_high: float = 99.0,
    threshold: float = 0.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    torch.manual_seed(2024)
    np.random.seed(2024)
    if device.type == "cuda":
        torch.cuda.manual_seed(2024)

    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {vol.shape}")

    D, H, W = vol.shape
    print("Volume shape (D,H,W):", (D, H, W))

    vol_u8 = normalize_mri_to_uint8(vol, p_low=p_low, p_high=p_high)
    frames = resize_grayscale_stack_to_rgb(vol_u8, image_size)

    frames_t = torch.from_numpy(frames).to(device)
    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=device)[:, None, None]
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=device)[:, None, None]
    frames_t = (frames_t - img_mean) / img_std

    if hasattr(predictor, "to"):
        try:
            predictor = predictor.to(device)
        except Exception:
            pass

    if hasattr(predictor, "model") and hasattr(predictor.model, "to"):
        try:
            predictor.model = predictor.model.to(device)
        except Exception:
            pass

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    segs_3d = np.zeros((D, H, W), dtype=np.uint8)

    def has_valid_prompt(points, point_labels, bbox, mask_input):
        has_points = points is not None and point_labels is not None and len(points) > 0
        has_box = bbox is not None
        has_mask = mask_input is not None and mask_input.sum() > 0
        return has_points or has_box or has_mask

    def add_prompt_for_slice(inference_state, slice_idx, points, point_labels, bbox, mask_input):
        has_points = points is not None and point_labels is not None and len(points) > 0
        has_box = bbox is not None
        has_mask = mask_input is not None and mask_input.sum() > 0

        print(f"Adding prompt(s) on slice {slice_idx}")

        if has_mask:
            if not hasattr(predictor, "add_new_mask"):
                raise AttributeError(
                    "This predictor does not expose add_new_mask(...), so dense mask prompting is not available."
                )

            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=slice_idx,
                obj_id=1,
                mask=mask_input,
            )

        if has_points and has_box:
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=slice_idx,
                obj_id=1,
                points=points,
                labels=point_labels,
                box=bbox,
            )
        elif has_points:
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=slice_idx,
                obj_id=1,
                points=points,
                labels=point_labels,
            )
        elif has_box:
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=slice_idx,
                obj_id=1,
                box=bbox,
            )
        elif not has_mask:
            raise ValueError(f"No valid prompts found for slice {slice_idx}")

    valid_slice_indices = [
        slice_idx
        for slice_idx in sorted(prompts_by_slice.keys())
        if has_valid_prompt(*prompts_by_slice[slice_idx])
    ]

    if len(valid_slice_indices) == 0:
        raise ValueError("No usable prompts found on any slice.")

    with torch.inference_mode(), autocast_ctx:
        inference_state = predictor.init_state(frames_t, H, W)

        print("Using prompts from slices:", valid_slice_indices)

        for slice_idx in valid_slice_indices:
            points, point_labels, bbox, mask_input = prompts_by_slice[slice_idx]
            add_prompt_for_slice(inference_state, slice_idx, points, point_labels, bbox, mask_input)

        print("Forward propagation...")
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            mask2d = (out_mask_logits[0] > threshold).detach().cpu().numpy()[0].astype(np.uint8)
            segs_3d[out_frame_idx] = np.maximum(segs_3d[out_frame_idx], mask2d)

        print("Backward propagation...")
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, reverse=True
        ):
            mask2d = (out_mask_logits[0] > threshold).detach().cpu().numpy()[0].astype(np.uint8)
            segs_3d[out_frame_idx] = np.maximum(segs_3d[out_frame_idx], mask2d)

        predictor.reset_state(inference_state)

    return segs_3d


def save_mask_like_reference(mask_3d: np.ndarray, reference_itk, output_mask: str):
    mask_itk = sitk.GetImageFromArray(mask_3d.astype(np.uint8))
    mask_itk.CopyInformation(reference_itk)

    out_dir = os.path.dirname(os.path.abspath(output_mask))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sitk.WriteImage(mask_itk, output_mask)
    print("Saved mask to:", output_mask)


def run_gui():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_nii", type=str, required=True, help="Path to MRI NIfTI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/MedSAM2_latest.pt")
    parser.add_argument("--cfg", type=str, default="configs/sam2.1_hiera_t512.yaml")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--p_low", type=float, default=1.0)
    parser.add_argument("--p_high", type=float, default=99.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--output_mask", type=str, default="", help="Optional output path for saved mask")
    args = parser.parse_args()

    print("Loading image...")
    img_itk = sitk.ReadImage(args.input_nii)
    vol = sitk.GetArrayFromImage(img_itk)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {vol.shape}")

    vol_u8 = normalize_mri_to_uint8(vol, p_low=args.p_low, p_high=args.p_high)

    print("Building predictor...")
    predictor = build_sam2_video_predictor_npz(args.cfg, args.checkpoint)
    print("Has add_new_mask:", hasattr(predictor, "add_new_mask"))

    viewer = napari.Viewer()
    viewer.add_image(vol_u8, name="image")

    inspect_layer = viewer.add_labels(
        np.zeros_like(vol_u8, dtype=np.uint16),
        name="inspect_labels",
    )

    pos_layer = viewer.add_points(
        name="positive",
        ndim=3,
        size=8,
        face_color="lime",
    )

    neg_layer = viewer.add_points(
        name="negative",
        ndim=3,
        size=8,
        face_color="red",
    )

    box_layer = viewer.add_shapes(
        name="bbox",
        ndim=3,
        shape_type="rectangle",
        edge_color="cyan",
    )

    mask_prompt_layer = viewer.add_labels(
        np.zeros_like(vol_u8, dtype=np.uint8),
        name="mask_prompt",
    )

    mask_layer = viewer.add_labels(
        np.zeros_like(vol_u8, dtype=np.uint8),
        name="mask",
    )

    @magicgui(
        call_button="Load dense mask prompt",
        mask_path={"label": "Mask path", "value": ""},
    )
    def load_mask_prompt(mask_path: str = ""):
        try:
            if mask_path is None or mask_path.strip() == "":
                show_error("Please provide a mask path.")
                return

            loaded_mask = load_mask_like_reference(mask_path, vol.shape)
            mask_prompt_layer.data = loaded_mask
            show_info(f"Loaded dense mask prompt from {mask_path}")

        except Exception as e:
            show_error(str(e))
            raise

    @magicgui(
        call_button="Run MedSAM2",
        auto_call=False,
        output_path={"label": "Save mask to", "value": args.output_mask},
        auto_box_from_mask={"label": "Auto-generate box prompts from segmentation", "value": True},
        auto_box_pad_px={"label": "Box padding (pixels)", "value": 5},
        auto_box_pad_frac={"label": "Box padding (fraction)", "value": 0.0},
        keep_largest_component={"label": "Keep largest 3D connected component", "value": True},
        require_prompted_slice_connection={"label": "Require connection to prompted slices", "value": False},
    )
    def run_segmentation(
        output_path: str = "",
        auto_box_from_mask: bool = True,
        auto_box_pad_px: int = 5,
        auto_box_pad_frac: float = 0.0,
        keep_largest_component: bool = True,
        require_prompted_slice_connection: bool = False,
    ):
        try:
            prompted_slices = get_all_prompted_slices(
                pos_layer, neg_layer, box_layer, mask_prompt_layer
            )

            if len(prompted_slices) == 0:
                show_error("No prompts found. Add points, draw a box, or load/draw a mask first.")
                return

            prompts_by_slice = {}
            for slice_idx in prompted_slices:
                points, point_labels, bbox, mask_input = collect_prompts_for_slice(
                    slice_idx, pos_layer, neg_layer, box_layer, mask_prompt_layer
                )
                prompts_by_slice[slice_idx] = (points, point_labels, bbox, mask_input)

            print("Using prompted slices:", prompted_slices)
            for s in prompted_slices:
                pts, lbls, bb, mask_input = prompts_by_slice[s]
                print(f"slice {s}:")
                print("  points:", pts)
                print("  labels:", lbls)
                print("  bbox:", bb)
                print("  mask prompt present:", mask_input is not None)

            segs_3d = run_medsam2_inference_from_arrays(
                vol=vol,
                predictor=predictor,
                image_size=args.image_size,
                prompts_by_slice=prompts_by_slice,
                p_low=args.p_low,
                p_high=args.p_high,
                threshold=args.threshold,
            )

            if keep_largest_component:
                if require_prompted_slice_connection:
                    segs_3d = keep_largest_3d_component_touching_prompted_slices(
                        segs_3d,
                        prompted_slices=prompted_slices,
                    )
                else:
                    segs_3d = keep_largest_3d_connected_component(segs_3d)

            mask_layer.data = segs_3d.astype(np.uint8)


            if auto_box_from_mask:
                auto_shapes = generate_bbox_shapes_from_mask(
                    segs_3d,
                    pad_px=auto_box_pad_px,
                    pad_frac=auto_box_pad_frac,
                )
                box_layer.data = auto_shapes
                show_info(
                    f"Segmentation finished using prompts from slices: {prompted_slices}. "
                    f"Generated {len(auto_shapes)} box prompts from the mask."
                )
            else:
                show_info(f"Segmentation finished using prompts from slices: {prompted_slices}")

            if output_path is not None and output_path.strip() != "":
                save_mask_like_reference(segs_3d, img_itk, output_path)
                show_info(f"Saved mask to {output_path}")

        except Exception as e:
            show_error(str(e))
            raise

    @magicgui(
        call_button="Save segmentation",
        save_path={"label": "Save path", "value": args.output_mask},
    )
    def save_current_segmentation(save_path: str = ""):
        try:
            if save_path is None or save_path.strip() == "":
                show_error("Please provide a save path.")
                return

            current_mask = np.asarray(mask_layer.data > 0, dtype=np.uint8)

            if current_mask.sum() == 0:
                show_error("Current segmentation is empty.")
                return

            save_mask_like_reference(current_mask, img_itk, save_path)
            show_info(f"Saved segmentation to {save_path}")

        except Exception as e:
            show_error(str(e))
            raise

    push_to_inspect_btn = PushButton(text="Push to inspect layer")

    @push_to_inspect_btn.clicked.connect
    def _push_segmentation_to_inspect():
        try:
            current_mask = np.asarray(mask_layer.data > 0, dtype=np.uint8)

            if current_mask.sum() == 0:
                show_error("Current segmentation is empty.")
                return

            inspect_data = np.asarray(inspect_layer.data).copy()
            next_label = int(inspect_data.max()) + 1

            inspect_data[current_mask > 0] = next_label

            inspect_layer.data = inspect_data

            # optional but recommended
            mask_layer.data = np.zeros_like(mask_layer.data, dtype=np.uint8)

            show_info(f"Pushed to inspect layer as label {next_label}")

        except Exception as e:
            show_error(str(e))
            raise



    reset_seg_btn = PushButton(text="Reset segmentation")
    reset_box_btn = PushButton(text="Reset box prompts")
    reset_dense_btn = PushButton(text="Reset dense prompts")
    reset_points_btn = PushButton(text="Reset positive + negative prompts")
    reset_inspect_btn = PushButton(text="Reset inspect layer")

    @reset_seg_btn.clicked.connect
    def _reset_segmentation():
        try:
            mask_layer.data = np.zeros_like(mask_layer.data, dtype=np.uint8)
            show_info("Segmentation reset.")
        except Exception as e:
            show_error(str(e))
            raise


    @reset_box_btn.clicked.connect
    def _reset_boxes():
        try:
            box_layer.data = []
            show_info("Box prompts reset.")
        except Exception as e:
            show_error(str(e))
            raise


    @reset_dense_btn.clicked.connect
    def _reset_dense():
        try:
            mask_prompt_layer.data = np.zeros_like(mask_prompt_layer.data, dtype=np.uint8)
            show_info("Dense prompts reset.")
        except Exception as e:
            show_error(str(e))
            raise


    @reset_points_btn.clicked.connect
    def _reset_points():
        try:
            pos_layer.data = np.empty((0, 3), dtype=np.float32)
            neg_layer.data = np.empty((0, 3), dtype=np.float32)
            show_info("Positive and negative prompts reset.")
        except Exception as e:
            show_error(str(e))
            raise

    @reset_inspect_btn.clicked.connect
    def _reset_inspect():
        try:
            inspect_layer.data = np.zeros_like(inspect_layer.data, dtype=np.uint16)
            show_info("Inspect layer reset.")
        except Exception as e:
            show_error(str(e))
            raise

    manage_segmentation_panel = Container(
        widgets=[
            save_current_segmentation,   # magicgui widget
            push_to_inspect_btn,
            reset_seg_btn,
            reset_box_btn,
            reset_dense_btn,
            reset_points_btn,
            reset_inspect_btn,
        ],
        labels=False,
        )


    viewer.window.add_dock_widget(load_mask_prompt, area="right")
    viewer.window.add_dock_widget(run_segmentation, area="right")
    viewer.window.add_dock_widget(manage_segmentation_panel, area="right", name="Manage segmentation")
    napari.run()


if __name__ == "__main__":
    run_gui()