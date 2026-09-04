# DICOM reader

`dicom_reader` is a small Python utility for reading 3D DICOM image stacks and converting RTSTRUCT contours into voxel masks. It is designed for working with CT/PET/MRI volumes and ROI data exported from DICOM. If your dataset does not include RTSTRUCT information, consider using [SimpleITK](https://simpleitk.org/) instead.

## Features

- Read a DICOM slice directory into a structured `DICOMImage` object
- Sort slices by patient position and geometry
- Access metadata such as spacing, origin, orientation, and manufacturer details
- Convert RTSTRUCT ROI contours into 3D binary masks via `DICOMStruct`
- Read SEG files through `DICOMSeg`
- Extract pixel arrays for downstream processing or visualization

## Project structure

```text
dicom_reader/
├── dicom_reader.py      # Main DICOM processing module
├── example.py           # Runnable example script for loading ROI masks
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
├── LICENSE              # Project license
└── .gitignore           # Repository ignores (if present)
```

## Installation

Clone the repository and install the project dependencies from the included requirements file:

```bash
git clone https://github.com/LangDaniel/dicom_reader
cd dicom_reader
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

The example script is meant to be edited directly with your local DICOM paths and then run as a simple Python script:

```bash
python3 example.py
```

The following example loads a CT stack and converts one ROI from an RTSTRUCT file into a 3D mask.

```python
import dicom_reader

# Example DICOM paths
img_dir = "/path/to/ct/slices"
rtstruct_file = "/path/to/rtstruct.dcm"

# Load image data
img_ds = dicom_reader.DICOMImage(img_dir)
img_array = img_ds.get_pixel_array() # numpy image voxel data

# Load the ROI structure and select one contour
struct_ds = dicom_reader.DICOMStruct(
    rtstruct_file,
    origin=img_ds.get_origin(),
    spacing=img_ds.get_spacing(),
    shape=img_ds.get_shape(),
)

# select the ROI
struct_ds.set_ROI_idx("GTV-1") 
roi_mask = struct_ds.get_pixel_array() # numpy ROI voxel data
```

## Notes

- The module expects DICOM files to be organized in a folder of slices.
- The contour conversion assumes the RTSTRUCT is aligned with the image volume metadata.
- `DICOMImage.get_pixel_array()` applies rescaling for CT and PET images using `RescaleSlope` and `RescaleIntercept`.
- This project is focused on DICOM mask extraction rather than a full medical imaging framework.


## License

This project is released under the MIT License.
