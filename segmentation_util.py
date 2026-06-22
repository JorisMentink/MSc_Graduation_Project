from contextlib import nullcontext
from PIL import Image
import numpy as np
import torch

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

def run_medsam2_inference_from_arrays(
    vol: np.ndarray,
    predictor,
    image_size: int,
    prompts_by_slice: dict[int, tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]],
    p_low: float = 1.0,
    p_high: float = 99.0,
    threshold: float = 0.0,
    propagation_style: str = "default",
    nr_propagation_slices: int = 2,
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
    #For returning logits together with thresholded segmentation.
    logits_3d = np.zeros((D, H, W), dtype=np.float32)
    logit_count = np.zeros((D, H, W), dtype=np.float32)

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

    def unpack_prompt(prompt):
        "Function to unpack prompt dicts to boolean format, compatible with rest of inference pipeline"
        
        if isinstance(prompt, dict):
            return (
                prompt.get("points", None),
                prompt.get("point_labels", None),
                prompt.get("bbox", None),
                prompt.get("mask_input", None),
            )

        return prompt

    valid_slice_indices = [
        slice_idx
        for slice_idx in sorted(prompts_by_slice.keys())
        if has_valid_prompt(*unpack_prompt(prompts_by_slice[slice_idx]))
    ]

    if len(valid_slice_indices) == 0:
        raise ValueError("No usable prompts found on any slice.")

    with torch.inference_mode(), autocast_ctx:
        inference_state = predictor.init_state(frames_t, H, W)

        print("Using prompts from slices:", valid_slice_indices)

        for slice_idx in valid_slice_indices:
            points, point_labels, bbox, mask_input = unpack_prompt(prompts_by_slice[slice_idx])
            add_prompt_for_slice(inference_state, slice_idx, points, point_labels, bbox, mask_input)

        # Propagation strategies:

        #Default propagation: propagate forward and backward from the first prompted slice.
        if propagation_style == "default":
            print("Forward propagation (default)...")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                
                logit2d = out_mask_logits[0].detach().cpu().numpy()[0].astype(np.float32)
                logits_3d[out_frame_idx] += logit2d
                logit_count[out_frame_idx] += 1

            print("Backward propagation (default)...")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):

                logit2d = out_mask_logits[0].detach().cpu().numpy()[0].astype(np.float32)
                logits_3d[out_frame_idx] += logit2d
                logit_count[out_frame_idx] += 1
            
            #Combine logits via averaging if relevant.
            valid = logit_count > 0
            logits_3d[valid] = logits_3d[valid] / logit_count[valid]
            segs_3d = (logits_3d > threshold).astype(np.uint8)

        #Full propagation: do a full forward and backward pass of propagation.
        elif propagation_style == "full":
            print("Forward propagation (full, from slice 0)...")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=0):
                
                logit2d = out_mask_logits[0].detach().cpu().numpy()[0].astype(np.float32)
                logits_3d[out_frame_idx] += logit2d
                logit_count[out_frame_idx] += 1

            print(f"Backward propagation (full, from slice {D - 1})...")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=D - 1, reverse=True):
                
                logit2d = out_mask_logits[0].detach().cpu().numpy()[0].astype(np.float32)
                logits_3d[out_frame_idx] += logit2d
                logit_count[out_frame_idx] += 1

            #Combine logits via averaging if relevant.
            valid = logit_count > 0
            logits_3d[valid] = logits_3d[valid] / logit_count[valid]
            segs_3d = (logits_3d > threshold).astype(np.uint8)

        #Smart propagation: propagate forward from the first prompted slice and backwards from the last slice
        elif propagation_style == "prompt_based":
            
            start_fwd = min(valid_slice_indices) # first prompted slice
            start_bwd = max(valid_slice_indices) # last prompted slice
            
            print(f"Forward propagation (prompt_based, from slice {start_fwd})...")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=start_fwd):
                
                logit2d = out_mask_logits[0].detach().cpu().numpy()[0].astype(np.float32)
                logits_3d[out_frame_idx] += logit2d
                logit_count[out_frame_idx] += 1

            print(f"Backward propagation (prompt_based, from slice {start_bwd})...")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=start_bwd, reverse=True):
                
                logit2d = out_mask_logits[0].detach().cpu().numpy()[0].astype(np.float32)
                logits_3d[out_frame_idx] += logit2d
                logit_count[out_frame_idx] += 1
                
            #Combine logits via averaging if relevant.
            valid = logit_count > 0
            logits_3d[valid] = logits_3d[valid] / logit_count[valid]
            segs_3d = (logits_3d > threshold).astype(np.uint8)
            
        
        #Central propagation: propagate forward and backward from a central slice (e.g. middle slice or middle prompted slice).
        elif propagation_style == "central_start":
            
            start_mid = sorted(valid_slice_indices)[len(valid_slice_indices) // 2]

            print(f"Forward propagation (from middle slice {start_mid})...")

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,start_frame_idx=start_mid):
                
                logit2d = out_mask_logits[0].detach().cpu().numpy()[0].astype(np.float32)
                logits_3d[out_frame_idx] += logit2d
                logit_count[out_frame_idx] += 1

            print(f"Backward propagation (from middle slice {start_mid})...")

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,start_frame_idx=start_mid,reverse=True):
                
                logit2d = out_mask_logits[0].detach().cpu().numpy()[0].astype(np.float32)
                logits_3d[out_frame_idx] += logit2d
                logit_count[out_frame_idx] += 1

            #Combine logits via averaging if relevant.
            valid = logit_count > 0
            logits_3d[valid] = logits_3d[valid] / logit_count[valid]
            segs_3d = (logits_3d > threshold).astype(np.uint8)

        #central_partitions propagation: select multiple central slices (e.g. 2 or 3) as start points for propagation, by selecting the most central prompted slices. Propagate forward and backward from each of these central slices and combine results (e.g. by taking union of predicted masks).
        elif propagation_style == "central_partitions":

            valid_slice_indices = sorted(valid_slice_indices)


            if nr_propagation_slices < 1:
                raise ValueError(f"Number of propagation slices must be at least 1, got {nr_propagation_slices}.")
            elif nr_propagation_slices > len(valid_slice_indices):
                print(
                    f"Requested {nr_propagation_slices} start slices, but only "
                    f"{len(valid_slice_indices)} prompted slices are available. "
                    f"Using all prompted slices instead."
                )
                start_slices = valid_slice_indices
            else:
                
                positions = np.linspace(0,len(valid_slice_indices) - 1,nr_propagation_slices + 2)[1:-1]
                selected_indices = [valid_slice_indices[int(round(pos))]for pos in positions]
                start_slices = sorted(set(selected_indices))
                print(f"Using central partition start slices: {start_slices}")

            logit_sum = np.zeros((D, H, W), dtype=np.float32)
            logit_count = np.zeros((D, H, W), dtype=np.float32)

            for start_slice in start_slices:

                print(f"Forward propagation from slice {start_slice}...")
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=start_slice
                ):
                    logit2d = out_mask_logits[0].detach().cpu().numpy()[0].astype(np.float32)
                    logit_sum[out_frame_idx] += logit2d
                    logit_count[out_frame_idx] += 1

                print(f"Backward propagation from slice {start_slice}...")
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=start_slice,
                    reverse=True
                ):
                    logit2d = out_mask_logits[0].detach().cpu().numpy()[0].astype(np.float32)
                    logit_sum[out_frame_idx] += logit2d
                    logit_count[out_frame_idx] += 1

            valid_logits = logit_count > 0

            avg_logits = np.zeros_like(logit_sum)
            avg_logits[valid_logits] = logit_sum[valid_logits] / logit_count[valid_logits]

            logits_3d = avg_logits
            segs_3d = (logits_3d > threshold).astype(np.uint8)

        else:
            raise ValueError(f"Unknown propagation_style '{propagation_style}'. Choose from: 'default', 'full', 'prompt_based', 'central_start', 'central_partitions'.")

        predictor.reset_state(inference_state)

    return segs_3d, logits_3d

def combine_prompt_sets(prompt_dict_list):
    """
    Combine multiple prompt dictionaries into a single prompt dictionary.

    Parameters
    ----------
    prompt_dict_list : list[dict]
        List of prompt dictionaries in the format:
        {
            z: {
                "points": np.ndarray or None,
                "point_labels": np.ndarray or None,
                "bbox": np.ndarray or None,
                "mask_input": np.ndarray or None,
            }
        }

    Returns
    -------
    combined : dict
        Combined prompt dictionary in the same format.
    """

    combined = {}

    for prompt_dict in prompt_dict_list:

        for z, prompt in prompt_dict.items():

            # Create empty slice entry if needed
            if z not in combined:
                combined[z] = {
                    "points": [],
                    "point_labels": [],
                    "bbox": None,
                    "mask_input": None,
                }

            # Collect points
            if prompt.get("points") is not None:
                combined[z]["points"].append(prompt["points"])

            if prompt.get("point_labels") is not None:
                combined[z]["point_labels"].append(prompt["point_labels"])

            # Keep the last bbox encountered
            if prompt.get("bbox") is not None:
                combined[z]["bbox"] = prompt["bbox"]

            # Union mask inputs
            if prompt.get("mask_input") is not None:
                if combined[z]["mask_input"] is None:
                    combined[z]["mask_input"] = prompt["mask_input"].copy()
                else:
                    combined[z]["mask_input"] = np.maximum(
                        combined[z]["mask_input"],
                        prompt["mask_input"]
                    )

    # Convert lists of points into arrays
    for z in combined:

        if len(combined[z]["points"]) > 0:
            combined[z]["points"] = np.concatenate(
                combined[z]["points"], axis=0
            )
            combined[z]["point_labels"] = np.concatenate(
                combined[z]["point_labels"], axis=0
            )
        else:
            combined[z]["points"] = None
            combined[z]["point_labels"] = None

    return combined

def compile_prompt_sets(prompt_dict_list, prompt_set_names, prompt_set_weights = None):

    prompt_sets_dict = {}

    for idx, name in enumerate(prompt_set_names):
        
        prompt_sets_dict[name] = (prompt_dict_list[idx],prompt_set_weights[idx])

    return prompt_sets_dict