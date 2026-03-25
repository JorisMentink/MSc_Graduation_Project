import os
import argparse
import numpy as np
import SimpleITK as sitk
import napari
from magicgui import magicgui
from magicgui.widgets import PushButton, Container
from napari.utils.notifications import show_info, show_error

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
        auto_dense_prompt_from_mask={"label": "Auto-generate dense prompt from segmentation", "value": False},
        keep_largest_component={"label": "Keep largest 3D connected component", "value": True},
        require_prompted_slice_connection={"label": "Require connection to prompted slices", "value": False},
    )
    def run_segmentation(
        output_path: str = "",
        auto_box_from_mask: bool = True,
        auto_box_pad_px: int = 5,
        auto_box_pad_frac: float = 0.0,
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
                    pad_px=auto_box_pad_px,
                    pad_frac=auto_box_pad_frac,
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