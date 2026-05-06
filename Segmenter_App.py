import os
import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import napari
from magicgui import magicgui
from magicgui.widgets import PushButton, Container
from napari.utils.notifications import show_info, show_error
from napari.utils.colormaps import DirectLabelColormap

from sam2.build_sam import build_sam2_video_predictor_npz

from App_utils.inference import run_medsam2_inference_from_arrays
from App_utils.prompt_utils import (
    load_mask_like_reference,
    generate_bbox_shapes_from_mask,
    collect_prompts_for_slice,
    get_all_prompted_slices,
)
from App_utils.pre_processing import normalize_mri_to_uint8
from App_utils.post_processing import (
    keep_largest_3d_connected_component,
    keep_largest_3d_component_touching_prompted_slices,
    save_mask_like_reference,
)


def get_image_path_from_folder(input_folder: str) -> str:
    image_path = os.path.join(input_folder, "image.nii.gz")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Could not find image.nii in folder: {input_folder}"
        )
    return image_path

def get_available_mask_files(input_folder: str) -> list[str]:
    if not os.path.isdir(input_folder):
        return []

    nii_files = []
    for fname in os.listdir(input_folder):
        is_nii = fname.lower().endswith(".nii") or fname.lower().endswith(".nii.gz")
        if is_nii and fname not in ("image.nii", "image.nii.gz"):
            nii_files.append(fname)

    nii_files.sort()
    return nii_files

def run_gui():
    parser = argparse.ArgumentParser()
    folder_group = parser.add_mutually_exclusive_group()
    folder_group.add_argument(
        "--input_folder",
        type=str,
        default=None,
        help="Folder containing image.nii and optional mask .nii files",
    )
    folder_group.add_argument(
        "--subject_index",
        type=int,
        default=None,
        help="Subject index into data/LUNDPROBE/ExtendedSamples (integer).",
    )
    folder_group.add_argument(
        "--subject_id",
        type=str,
        default=None,
        help="Subject folder name inside data/LUNDPROBE/ExtendedSamples.",
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/MedSAM2_latest.pt")
    parser.add_argument("--cfg", type=str, default="configs/sam2.1_hiera_t512.yaml")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--p_low", type=float, default=1.0)
    parser.add_argument("--p_high", type=float, default=99.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--output_mask", type=str, default="", help="Optional output path for saved mask")
    args = parser.parse_args()

    split_dir = Path("data") / "LUNDPROBE" / "ExtendedSamples" / "development"
    subjects = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])

    if args.input_folder is None and args.subject_index is None and args.subject_id is None:
        print("Available subjects (pass one of these):")
        for i, name in enumerate(subjects):
            print(f"  --subject_index {i:>3d}   --subject_id {name}")
        parser.exit(0)

    if args.subject_id is not None:
        if args.subject_id not in subjects:
            parser.error(f"Subject '{args.subject_id}' not found. Run without arguments to list all subjects.")
        input_folder = str(split_dir / args.subject_id / "MR_StorT2")
    elif args.subject_index is not None:
        input_folder = str(split_dir / subjects[args.subject_index] / "MR_StorT2")
    else:
        input_folder = args.input_folder
    image_path = get_image_path_from_folder(input_folder)
    available_mask_files = get_available_mask_files(input_folder)

    print("Loading image...")
    img_itk = sitk.ReadImage(image_path)
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
        size=4,
        face_color="lime",
    )

    neg_layer = viewer.add_points(
        name="negative",
        ndim=3,
        size=4,
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
        opacity=0.55,
    )
    mask_prompt_layer.colormap = DirectLabelColormap(
        color_dict={
            0: (0, 0, 0, 0),            # transparent
            1: (0.20, 0.45, 0.85, 1.0), # muted blue
        }
    )

    gt_layer = viewer.add_labels(
        np.zeros_like(vol_u8, dtype=np.uint8),
        name="ground_truth",
        opacity=0.60,
    )
    gt_layer.colormap = DirectLabelColormap(
        color_dict={
            0: (0, 0, 0, 0),            # transparent
            1: (0.18, 0.62, 0.38, 1.0), # dark soft green
        }
    )

    mask_layer = viewer.add_labels(
        np.zeros_like(vol_u8, dtype=np.uint8),
        name="mask",
        opacity=0.60,
    )
    mask_layer.colormap = DirectLabelColormap(
        color_dict={
            0: (0, 0, 0, 0),            # transparent
            1: (0.78, 0.28, 0.28, 1.0), # dark soft red
        }
    )


    available_prompt_files = sorted(
        f for f in os.listdir(input_folder) if f.lower().endswith(".npz")
    )

    @magicgui(
        call_button="Load point prompts (.npz)",
        prompt_file={
            "label": "Prompt file",
            "choices": [""] + available_prompt_files,
        },
        replace_existing={"label": "Replace existing points", "value": True},
    )
    def load_point_prompts(prompt_file: str = "", replace_existing: bool = True):
        try:
            if not prompt_file or prompt_file.strip() == "":
                show_error("Please select a prompt file.")
                return
            path = os.path.join(input_folder, prompt_file)
            data = np.load(path)
            pos_pts = data["positive"].astype(np.float32) if "positive" in data else np.empty((0, 3), dtype=np.float32)
            neg_pts = data["negative"].astype(np.float32) if "negative" in data else np.empty((0, 3), dtype=np.float32)
            if replace_existing:
                pos_layer.data = pos_pts
                neg_layer.data = neg_pts
            else:
                existing_pos = np.asarray(pos_layer.data, dtype=np.float32)
                existing_neg = np.asarray(neg_layer.data, dtype=np.float32)
                pos_layer.data = np.concatenate([existing_pos, pos_pts], axis=0) if len(existing_pos) else pos_pts
                neg_layer.data = np.concatenate([existing_neg, neg_pts], axis=0) if len(existing_neg) else neg_pts
            show_info(f"Loaded {len(pos_pts)} positive and {len(neg_pts)} negative prompts from {prompt_file}")
        except Exception as e:
            show_error(str(e))
            raise

    @magicgui(
        call_button="Load dense mask prompt",
        mask_file={
            "label": "Mask file",
            "choices": [""] + available_mask_files,
        },
    )
    def load_mask_prompt(mask_file: str = ""):
        try:
            if mask_file is None or mask_file.strip() == "":
                show_error("Please select a mask file.")
                return

            mask_path = os.path.join(input_folder, mask_file)
            loaded_mask = load_mask_like_reference(mask_path, vol.shape)
            mask_prompt_layer.data = loaded_mask
            show_info(f"Loaded dense mask prompt from {mask_file}")

        except Exception as e:
            show_error(str(e))
            raise

    @magicgui(
        call_button="Run MedSAM2",
        auto_call=False,
        propagation_style={
            "label": "Propagation method",
            "choices": [
                ("Default (prompt-first)", "default"),
                ("Full (entire volume)", "full"),
                ("Smart (prompt-based)", "prompt_based"),
            ],
            "value": "default",
        },
        auto_box_from_mask={"label": "Auto-generate box prompts from segmentation", "value": True},
        auto_dense_prompt_from_mask={"label": "Auto-generate dense prompt from segmentation", "value": False},
        keep_largest_component={"label": "Keep largest 3D connected component", "value": True},
        require_prompted_slice_connection={"label": "Require connection to prompted slices", "value": False},
    )
    def run_segmentation(
        output_path: str = "",
        propagation_style: str = "default",
        auto_box_from_mask: bool = True,
        auto_dense_prompt_from_mask: bool = False,
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
                propagation_style=propagation_style,
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

            if auto_dense_prompt_from_mask:
                mask_prompt_layer.data = segs_3d.astype(np.uint8)

            if auto_box_from_mask:
                box_layer.data = generate_bbox_shapes_from_mask(
                    segs_3d,
                )

            show_info("Segmentation updated.")

            if output_path is not None and output_path.strip() != "":
                save_mask_like_reference(segs_3d, img_itk, output_path)
                show_info(f"Saved mask to {output_path}")

        except Exception as e:
            show_error(str(e))
            raise

    @magicgui(
    call_button="Save segmentation",
    save_name={"label": "File name", "value": "prediction.nii.gz"},
    )
    def save_current_segmentation(save_name: str = "prediction.nii.gz"):
        try:
            if save_name is None or save_name.strip() == "":
                show_error("Please provide a file name.")
                return

            if not (save_name.lower().endswith(".nii") or save_name.lower().endswith(".nii.gz")):
                show_error("File name must end with .nii or .nii.gz")
                return

            current_mask = np.asarray(mask_layer.data > 0, dtype=np.uint8)

            if current_mask.sum() == 0:
                show_error("Current segmentation is empty.")
                return

            output_dir = os.path.join(input_folder, "SAM_OUTPUT")
            os.makedirs(output_dir, exist_ok=True)  # creates if not exists

            save_path = os.path.join(output_dir, save_name)
            
            save_mask_like_reference(current_mask, img_itk, save_path)
            show_info(f"Saved segmentation to {save_path}")

        except Exception as e:
            show_error(str(e))
            return

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

    @magicgui(
    call_button="Load ground truth",
    gt_file={
        "label": "Ground truth file",
        "choices": [""] + available_mask_files,
    },
    )
    def load_ground_truth(gt_file: str = ""):
        try:
            if gt_file is None or gt_file.strip() == "":
                show_error("Please select a ground truth file.")
                return

            gt_path = os.path.join(input_folder, gt_file)
            loaded_gt = load_mask_like_reference(gt_path, vol.shape)
            gt_layer.data = loaded_gt.astype(np.uint8)

            show_info(f"Loaded ground truth from {gt_file}")

        except Exception as e:
            show_error(str(e))
            raise

    manage_segmentation_panel = Container(
        widgets=[
            save_current_segmentation,
            push_to_inspect_btn,
            reset_seg_btn,
            reset_box_btn,
            reset_dense_btn,
            reset_points_btn,
            reset_inspect_btn,
        ],
        labels=False,
    )

    viewer.window.add_dock_widget(load_point_prompts, area="right")
    viewer.window.add_dock_widget(load_mask_prompt, area="right")
    viewer.window.add_dock_widget(run_segmentation, area="right")
    viewer.window.add_dock_widget(manage_segmentation_panel, area="right", name="Manage segmentation")
    viewer.window.add_dock_widget(load_ground_truth, area="right")
    napari.run()


if __name__ == "__main__":
    run_gui()