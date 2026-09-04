# API reference

This page summarizes the main classes and methods exposed by the project.

## DICOMImage

`DICOMImage` represents a 3D DICOM volume stored as a set of axial slice files inside a folder.

### Arguments

- **`dcm_folder`** - Path to the folder containing `*.dcm` files. All files in the folder are expected to belong to the same 3D image.

### Attributes

- **instance_uids**: NumPy array of sorted SOP Instance UIDs for the image slices.
- **slice_paths**: NumPy array containing the sorted slice file paths.

### Methods

- **`sort_instance_uids(slice_paths)`**: Sorts the slice files by patient height and returns the sorted instance UIDs and ordering indices.
- **`get_shape()`**: Returns the image shape as `(rows, columns, num_slices)`.
- **`get_orientation(precision=0.005)`**: Reads the patient orientation for the first slice and verifies that all slices share the same orientation within a tolerance.
- **`get_origin()`**: Returns the image origin from the first slice.
- **`get_slice_positions()`**: Returns the patient position for every slice as a 3xN array.
- **`get_pos_from_uid(uid)`**: Finds the patient position associated with a specific SOP Instance UID.
- **`get_spacing(precision=[0.05, 0.05, 0.05])`**: Computes the voxel spacing in x, y, and z and validates that the spacing is consistent.
    Raises a `ValueError` if the difference between the spacings of different slices is larger than `precision`.
- **`get_pixel_array()`**: Reads the pixel data from each slice and returns the full 3D image volume.
- **`get_manufacturer()`**: Returns the DICOM manufacturer name from the first slice.
- **`get_model()`**: Returns the manufacturer model name from the first slice.
- **`list_tags(slz=None)`**: Lists the available DICOM tags for a slice.
- **`get_tag(tag, slz=None)`**: Returns the value of the DICOM `tag` for a given slice.

## DICOMContour

`DICOMContour` is the abstract base class for DICOM segmentation objects such as RTSTRUCT and SEG data that belong to a given `DICOMImage`.
It is inherited by `DICOMStruct` and `DICOMSeg`.

### Arguments

-  **`file_path`** - Path to the DICOM segmentation object.
-  **`origin`** - Patient origin (to be retrieved from the corresponding `DICOMImage`).
-  **`spacing`** - Pixel spacing of the corresponding `DICOMImage`.
-  **`shape`** - Shape of the corresponding `DICOMImage`.

### Attributes

-  **ds** - `pydicom` instance of the segmentation file.

### Methods

- **`get_pixel_array()`**: Abstract property for subclasses to provide pixel data of the segmentation file.
- **`get_contour()`**: Abstract property for subclasses to provide contour point data.
- **`get_bbox(order='numpy')`**: Computes the bounding box of the segmentation using the requested axis ordering.
    The ordering should be one of `numpy`, `cv2`, `row_first`, or `col_first`, specifying the convention used.
- **`get_center(order='numpy')`**: Computes the center of mass of the segmentation region.

## DICOMSeg

`DICOMSeg(DICOMContour)` reads a DICOM SEG object and converts segmentation frames into a 3D binary mask.

### Methods

- **`check_axial_origin(precision=[0.05, 0.05])`**: Verifies that the SEG origin matches the image origin within the given precision.
- **`check_shape()`**: Validates the segmentation shape against the corresponding image volume.
- **`padding(data)`**: Pads the segmentation data into the full image volume when the SEG is smaller in the z dimension.

## DICOMStruct

`DICOMStruct(DICOMContour)` reads an RTSTRUCT file (`*.dcm`) and converts a selected ROI into a 3D binary mask.

### Attributes

-  **ROI_idx** - Index of the selected region of interest.

### Methods

- **`get_ROI_names()`**: Returns the list of available ROI names in the structure set.
- **`set_ROI_idx(ROI_name)`**: Selects the active ROI by name.
- **`get_ROI_index(ROI_name)`**: Finds the index of a specific ROI name.
- **`coordinates_to_pixel(coordinates)`**: Converts a physical scanner coordinate into its corresponding image voxel indices.

## Notes

- This project focuses on DICOM image loading and contour-to-mask conversion rather than full medical imaging workflows.
- ROI conversion assumes the source RTSTRUCT and the base image share a compatible geometry and spacing model.
- The implementation depends on `numpy`, `pydicom`, and `opencv-python` for array handling, DICOM parsing, and contour drawing.
