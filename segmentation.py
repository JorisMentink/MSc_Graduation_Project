from DataLoader import DataLoader
from sam2.build_sam import build_sam2_video_predictor_npz
from segmentation_util import run_medsam2_inference_from_arrays, compile_prompt_sets
import numpy as np

class Segmentation:
    def __init__(
        self,
        data: DataLoader,
        checkpoint="checkpoints/MedSAM2_latest.pt",
        cfg="configs/sam2.1_hiera_t512.yaml",
    ):
        if not isinstance(data, DataLoader):
            raise TypeError(
                "Segmentation expects a DataLoader instance. "
                "Please create one first: data = DataLoader(...)"
            )

        self.data = data

        self.parentfolder = data.parentfolder
        self.subject_nr = data.subject_nr
        self.subject_name = data.subject_name
        self.volume_of_interest = data.volume_of_interest
        self.verbose = data.verbose


        self.img = data.img
        self.mask = data.mask
        self.gt = data.gt
        self.img_spacing = data.img_spacing

        self.prompt_dict_list = []
        self.prompts_by_slice = {}

        if self.verbose:
            print(
                f"Initialized Segmentation for subject {self.subject_nr} "
                f"with volume of interest '{self.volume_of_interest}'"
            )
            print(f"Mask shape: {self.mask.shape}")
            print(f"Building SAM predictor from checkpoint: {checkpoint}")

        self.predictor = build_sam2_video_predictor_npz(cfg, checkpoint)

    def load_dense_prompt(self):
        "Function that loads existing dense mask as prompt"

        if self.mask is None or self.mask.sum() == 0:
            raise ValueError("No valid mask found to load as prompt.")

        dense_prompts = {}

        for z in range(self.mask.shape[0]):
            mask_2d = self.mask[z].astype(np.uint8)

            if mask_2d.sum() == 0:
                continue

            dense_prompts[z] = {
                "points": None,
                "point_labels": None,
                "bbox": None,
                "mask_input": mask_2d,
            }

        self.dense_prompt_set = dense_prompts

        return self.dense_prompt_set

    def check_loaded_prompts(self):
        """
        Prints summary of loaded prompts, including number of slices with prompts and their range.
        """

        print("\nPrompt summary")
        print("-" * 40)
        if hasattr(self, "compiled_prompt_sets") and self.compiled_prompt_sets is not None:
            print(f"Compiled prompt sets: {len(self.compiled_prompt_sets)}")

            for set_name, (prompt_dict, weight) in self.compiled_prompt_sets.items():
                slices = sorted(prompt_dict.keys())

                if len(slices) > 0:
                    print(f"  {set_name}: {len(slices)} slices "
                        f"({slices[0]} to {slices[-1]}), weight={weight}")
                else:
                    print(f"  {set_name}: 0 slices, weight={weight}")

        else:
            print("Compiled prompt sets: not created")

        print("-" * 40)


    def compile_prompt_sets(self, prompt_dict_list=None, prompt_set_names=None, prompt_set_weights=None):

        #If no prompt dict list is provided, this function defaults to use the dense mask as the only prompt set.
        if prompt_dict_list is None:
            
            if self.dense_prompt_set is None:
                raise ValueError("No prompt dict list provided and no dense prompt set found. Please run load_dense_prompt() first.")
            
            self.compiled_prompt_sets = { "dense_mask": (self.dense_prompt_set, 1.0) }
            return

        if prompt_set_weights is None:
            if not (len(prompt_dict_list) == len(prompt_set_names)):
                raise ValueError(f"prompt_dict_list and prompt_set_names must have the same length. Got {len(prompt_dict_list)} and {len(prompt_set_names)}.")
        else:
            if not (len(prompt_dict_list) == len(prompt_set_names) == len(prompt_set_weights)):
                raise ValueError(f"prompt_dict_list, prompt_set_names, and prompt_set_weights ""must all have the same length. Got {len(prompt_dict_list)}, {len(prompt_set_names)}, and {len(prompt_set_weights)}.")

        #Use np.isclose to account for potential float summing being funkyyy
        if not np.isclose(sum(prompt_set_weights), 1.0):
            raise ValueError("Sum of prompt set weights must be 1.0.")

        self.compiled_prompt_sets = compile_prompt_sets(prompt_dict_list, prompt_set_names, prompt_set_weights)
        return


    def remove_distant_slices(self, tolerance_frames=3):
        """
        Remove predicted segmentation slices that are more than tolerance_frames
        away from any slice containing dense mask pixels.
        """

        if not hasattr(self, "predicted_seg"):
            raise AttributeError(
                "No predicted segmentation found. Run run_segmentation() first."
            )

        if self.mask is None or self.mask.sum() == 0:
            raise ValueError("No valid dense mask found.")

        # Find z-slices where dense mask exists
        dense_slices = np.where(self.mask.astype(bool).any(axis=(1, 2)))[0]

        if len(dense_slices) == 0:
            raise ValueError("Dense mask contains no foreground slices.")

        z_min = max(0, dense_slices.min() - tolerance_frames)
        z_max = min(self.predicted_seg.shape[0] - 1, dense_slices.max() + tolerance_frames)

        # Create cleaned prediction
        cleaned_seg = np.zeros_like(self.predicted_seg)
        cleaned_seg[z_min:z_max + 1] = self.predicted_seg[z_min:z_max + 1]

        self.predicted_seg = cleaned_seg

        if self.verbose:
            print(
                f"Kept slices {z_min} to {z_max}. "
                f"Removed predicted segmentation outside dense mask ±{tolerance_frames} slices."
            )

        return self.predicted_seg


    def run_segmentation_sets(self, propagation_style="default", weighting_strategy="average", threshold=0.0):
        """Run segmentation for different prompt sets and combine results using specified logit fusion strategy.
        prompt sets are generated via split_prompts_to_sets, imported from segmentation_util.py"""

        if self.compiled_prompt_sets is None:
            raise ValueError("No compiled prompt sets found. Run compile_prompt_sets() first.")


        logits_per_set = []
        weights_used = []

        self.segs_per_set = {}

        #Loop through all prompt sets and run individual segmentations
        for num, (set_name, set_data) in enumerate(self.compiled_prompt_sets.items()):
            
            prompt_set, weight = set_data
            
            if weight is None or weight > 0.0:

                print(f"Running segmentation for prompt set '{set_name}' with slices: {list(prompt_set.keys())}")

                pred_seg, pred_logits = run_medsam2_inference_from_arrays(
                    vol=self.img,
                    predictor=self.predictor,
                    image_size=512,
                    prompts_by_slice=prompt_set,
                    p_low=1.0,
                    p_high=99.0,
                    threshold=threshold,
                    propagation_style=propagation_style,
                )

                self.segs_per_set[set_name] = pred_seg

                #Store logits for every prompt set for later fusion
                logits_per_set.append(pred_logits)

                if weight > 0.0:
                    weights_used.append(weight)

        #Basic, equal weighted logit averaging.
        if weighting_strategy == "average":
            final_logits = np.mean(logits_per_set, axis=0)

        elif weighting_strategy == "custom":
            #Creating empty frame to add all averaged logits to
            final_logits = np.zeros_like(logits_per_set[0], dtype=np.float32)

            for nr, weight_factor in enumerate(weights_used):
                final_logits += logits_per_set[nr] * weight_factor

        else:
            raise ValueError(f"Unknown weighting_strategy: {weighting_strategy}")

        self.predicted_seg = (final_logits > threshold).astype(np.uint8)
        self.predicted_logits = final_logits

        return self.predicted_seg, self.predicted_logits