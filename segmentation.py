from DataLoader import DataLoader
from sam2.build_sam import build_sam2_video_predictor_npz
from segmentation_util import run_medsam2_inference_from_arrays
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

        self.prompt_dict_list.append(dense_prompts)
        return dense_prompts

    def add_prompt_dict(self, prompt_dict):
        
        if not isinstance(prompt_dict, dict):
            raise TypeError("Prompt dict must be a dictionary with slice indices as keys.")

        self.prompt_dict_list.append(prompt_dict)

        if self.verbose:
            print(f"Added new prompt dict with slices: {list(prompt_dict.keys())}")


    def check_loaded_prompts(self):

        print(f"In total, {len(self.prompt_dict_list)} prompt dict(s) loaded.")
        for i, prompt_dict in enumerate(self.prompt_dict_list):
            print(f"{i}th prompt dict contains prompts for slices: {list(prompt_dict.keys())}")
        print(f"To combine prompts for segmentation, run construct_prompts")


    def construct_prompts(self):

        combined = {}

        for prompt_dict in self.prompt_dict_list:

            for z, prompt in prompt_dict.items():

                #Create empty slice for prompts to be appended to
                if z not in combined:
                    combined[z] = {
                        "points": [],
                        "point_labels": [],
                        "bbox": None,
                        "mask_input": None,
                    }

                #If point prompts are present, append their position and label to the list
                if prompt.get("points") is not None:
                    combined[z]["points"].append(prompt["points"])

                if prompt.get("point_labels") is not None:
                    combined[z]["point_labels"].append(prompt["point_labels"])

                #If a bbox prompt is present, use the lastly added one TODO: check if this is the best way
                if prompt.get("bbox") is not None:
                    combined[z]["bbox"] = prompt["bbox"]

                #If mask prompt is present, combine it with existing mask (union)
                if prompt.get("mask_input") is not None:
                    if combined[z]["mask_input"] is None:
                        combined[z]["mask_input"] = prompt["mask_input"]
                    else:
                        combined[z]["mask_input"] = np.maximum(
                            combined[z]["mask_input"],
                            prompt["mask_input"],
                        )

        #create final arrays
        for z in combined:
            # stack points
            if len(combined[z]["points"]) > 0:
                combined[z]["points"] = np.concatenate(combined[z]["points"], axis=0)
                combined[z]["point_labels"] = np.concatenate(combined[z]["point_labels"], axis=0)
            else:
                combined[z]["points"] = None
                combined[z]["point_labels"] = None

        #Store new prompts and return
        self.prompts_by_slice = combined
        self.prompt_dict_list = [combined]

        return self.prompts_by_slice


    def run_segmentation(self):
        self.predicted_seg = run_medsam2_inference_from_arrays(
            vol=self.img,
            predictor=self.predictor,
            image_size=512,
            prompts_by_slice=self.prompts_by_slice,
            p_low=1.0,
            p_high=99.0,
            threshold=0.0,
            propagation_style="default",
        )

        return self.predicted_seg
