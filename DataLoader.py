from pathlib import Path
import SimpleITK as sitk
from segmentation_util import compute_contour_average

class DataLoader():
    def __init__(self,parentfolder,subject_nr=0,volume_of_interest="CTVT",verbose=False,load_recontours=False):

        self.parentfolder = Path(parentfolder)
        self.subject_nr = subject_nr
        self.volume_of_interest = volume_of_interest
        self.verbose = verbose

        #Method only supports prostate volume (CTVT) and rectum as of now
        if self.volume_of_interest not in ["CTVT","rectum"]:
            raise ValueError("Volume of interest must be either 'CTVT' or 'rectum'")

        #Sorts subjects and selects subject folder based on subject number. Follows out-of-the-box LUNDPROBE formatting.
        self.subjects = sorted([p.name for p in self.parentfolder.iterdir() if p.is_dir()])
        self.subjectfolder = self.parentfolder / str(self.subjects[subject_nr]) / "MR_StorT2"
        
        self.subject_name = str(self.subjects[subject_nr])

        #Load paths for image, mask and uncertainty map -following out-of-the-box LUNDPROBE formatting
        img_path = self.subjectfolder / "image.nii.gz"
        
        if self.volume_of_interest == "CTVT":
            mask_path = self.subjectfolder / "nnUNetOutput/mask_CTVT_427_nnUNet.nii.gz"
            unc_path = self.subjectfolder / "nnUNetOutput/mask_CTVT_427_nnUNet_uncertaintyMap.nii.gz"
            gt_path = self.subjectfolder / "mask_CTVT_427.nii.gz"
                
        elif self.volume_of_interest == "rectum":
            mask_path = self.subjectfolder / "nnUNetOutput/mask_Rectum_nnUNet.nii.gz"
            unc_path = self.subjectfolder / "nnUNetOutput/mask_Rectum_nnUNet_uncertaintyMap.nii.gz"
            gt_path = self.subjectfolder / "mask_Rectum.nii.gz"
                
        #Load data as arrays
        self.img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
        self.mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) > 0
        self.unc_map = sitk.GetArrayFromImage(sitk.ReadImage(str(unc_path)))
        self.gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))) > 0
            
        #LOAD IMAGE SPACING
        img_itk = sitk.ReadImage(str(img_path))
        spacing_sitk = img_itk.GetSpacing()  # (x, y, z)
        self.img_spacing = spacing_sitk[::-1]  # (z, y, x)

        if self.verbose:
            print(f"Loaded subject {self.subjects[subject_nr]} with volume of interest '{self.volume_of_interest}'")
            print(f"Image shape: {self.img.shape}, Mask shape: {self.mask.shape}, Uncertainty map shape: {self.unc_map.shape}, Ground truth shape: {self.gt.shape}")
            print(f"Image spacing (z, y, x): {self.img_spacing}")
        
    def load_recontours(self):

        if self.volume_of_interest == "CTVT":
            observer_paths = sorted((self.subjectfolder / "observerData").glob("mask_CTVT_427_step2_obs*.nii.gz"))

        elif self.volume_of_interest == "rectum":
            observer_paths = sorted((self.subjectfolder / "observerData").glob("mask_Rectum_step2_obs*.nii.gz"))

        self.observer_recontours = [
            sitk.GetArrayFromImage(sitk.ReadImage(str(p))) > 0
            for p in observer_paths
        ]

        self.observer_names = [p.name.split("obs")[-1].replace(".nii.gz", "") for p in observer_paths]

        return self.observer_recontours
    
    def load_consensus(self,indices = [0,2,3]):
        """"
        Computes consensus of recontours for evaluation. Indices determines what recontours are used for consensus.
        Default value is [0,2,3] which corresponds to observers B, D and E. C was left out for possible cross validation.
        """

        if not hasattr(self, 'observer_recontours'):
            print("Load_recontours() was not executed yet. Executing now...")
            self.load_recontours()

        selected_recontours = [self.observer_recontours[i] for i in indices]
        self.consensus_recontour = compute_contour_average(selected_recontours, self.img_spacing)
        
        return self.consensus_recontour

    def load_summed_recontours(self,indices = [0,2,3]):
        """"
        Computes summed recontours for evaluation. Indices determines what recontours are used for summation.
        Default value is [0,2,3] which corresponds to observers B, D and E. C was left out for possible cross validation.
        """

        if not hasattr(self, 'observer_recontours'):
            print("Load_recontours() was not executed yet. Executing now...")
            self.load_recontours()

        selected_recontours = [self.observer_recontours[i] for i in indices]
        self.summed_recontour = sum(selected_recontours)
        
        return self.summed_recontour