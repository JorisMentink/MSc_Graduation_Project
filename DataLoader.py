from pathlib import Path
import SimpleITK as sitk
from pathlib import Path

class DataLoader():
    def __init__(self,parentfolder,subject_nr=0,volume_of_interest="CTVT",verbose=False):

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