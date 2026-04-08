from contextlib import nullcontext
import numpy as np
import torch

from App_utils.pre_processing import normalize_mri_to_uint8, resize_grayscale_stack_to_rgb


def run_medsam2_inference_from_arrays(
    vol: np.ndarray,
    predictor,
    image_size: int,
    prompts_by_slice: dict[int, tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]],
    p_low: float = 1.0,
    p_high: float = 99.0,
    threshold: float = 0.0,
    propagation_style: str = "default",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    torch.manual_seed(1604)
    np.random.seed(1604)
    
    if device.type == "cuda":
        torch.cuda.manual_seed(1604)

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
        predictor = predictor.to(device)

    if hasattr(predictor, "model") and hasattr(predictor.model, "to"):
        predictor.model = predictor.model.to(device)

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

        # Propagation strategies:

        #Default propagation: propagate forward and backward from the first prompted slice.
        if propagation_style == "default":
            print("Forward propagation (default)...")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                mask2d = (out_mask_logits[0] > threshold).detach().cpu().numpy()[0].astype(np.uint8)
                segs_3d[out_frame_idx] = np.maximum(segs_3d[out_frame_idx], mask2d)

            print("Backward propagation (default)...")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, reverse=True
            ):
                mask2d = (out_mask_logits[0] > threshold).detach().cpu().numpy()[0].astype(np.uint8)
                segs_3d[out_frame_idx] = np.maximum(segs_3d[out_frame_idx], mask2d)

        #Full propagation: do a full forward and backward pass of propagation.
        elif propagation_style == "full":
            print("Forward propagation (full, from slice 0)...")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=0
            ):
                mask2d = (out_mask_logits[0] > threshold).detach().cpu().numpy()[0].astype(np.uint8)
                segs_3d[out_frame_idx] = np.maximum(segs_3d[out_frame_idx], mask2d)

            print(f"Backward propagation (full, from slice {D - 1})...")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=D - 1, reverse=True
            ):
                mask2d = (out_mask_logits[0] > threshold).detach().cpu().numpy()[0].astype(np.uint8)
                segs_3d[out_frame_idx] = np.maximum(segs_3d[out_frame_idx], mask2d)

        #Smart propagation: propagate forward from the first prompted slice and backwards from the last slice
        elif propagation_style == "prompt_based":
            
            start_fwd = min(valid_slice_indices) # first prompted slice
            start_bwd = max(valid_slice_indices) # last prompted slice
            
            print(f"Forward propagation (prompt_based, from slice {start_fwd})...")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=start_fwd
            ):
                mask2d = (out_mask_logits[0] > threshold).detach().cpu().numpy()[0].astype(np.uint8)
                segs_3d[out_frame_idx] = np.maximum(segs_3d[out_frame_idx], mask2d)

            print(f"Backward propagation (prompt_based, from slice {start_bwd})...")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=start_bwd, reverse=True
            ):
                mask2d = (out_mask_logits[0] > threshold).detach().cpu().numpy()[0].astype(np.uint8)
                segs_3d[out_frame_idx] = np.maximum(segs_3d[out_frame_idx], mask2d)

        else:
            raise ValueError(f"Unknown propagation_style '{propagation_style}'. Choose from: 'default', 'full', 'smart'")

        predictor.reset_state(inference_state)

    return segs_3d