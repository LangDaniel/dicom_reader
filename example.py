import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba

import dicom_reader


IMG_DIR = "/path/to/ct/slices"
RTSTRUCT_FILE = "/path/to/rtstruct.dcm"
ROI_NAME = "GTV-1"
SLICE_INDEX = 35


def build_green_cmap():
    colors = [[1.0, 1.0, 1.0, 0.0]]
    colors.append(to_rgba("lime"))
    return ListedColormap(colors)


def prompt_for_roi_name(roi_names, default):
    print("Available ROIs:")
    for name in roi_names:
        print(f"  - {name}")

    raw_value = input(f"Select ROI name [{default}]: ").strip()
    value = raw_value if raw_value else default
    if value not in roi_names:
        raise ValueError(f"ROI '{value}' not found. Available ROIs: {roi_names}")
    return value


def prompt_for_slice_index(num_slices, default):
    start, end = 0, max(num_slices - 1, 0)
    print(f"Slice range: {start} to {end}")

    raw_value = input(f"Select slice index [{default}]: ").strip()
    value = int(raw_value) if raw_value else default
    if value < start or value > end:
        raise IndexError(
            f"Slice index {value} is out of range for image shape ({num_slices} slices)."
        )
    return value


def main():
    img_ds = dicom_reader.DICOMImage(IMG_DIR)
    img = img_ds.get_pixel_array()

    struct_ds = dicom_reader.DICOMStruct(
        RTSTRUCT_FILE,
        origin=img_ds.get_origin(),
        spacing=img_ds.get_spacing(),
        shape=img_ds.get_shape(),
    )

    roi_names = struct_ds.get_ROI_names()
    roi_name = prompt_for_roi_name(roi_names, ROI_NAME)

    struct_ds.set_ROI_idx(roi_name)
    roi_mask = struct_ds.get_pixel_array()

    slice_index = prompt_for_slice_index(img.shape[2], SLICE_INDEX)

    green_cmap = build_green_cmap()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(img[:, :, slice_index], cmap="bone")
    ax.imshow(roi_mask[:, :, slice_index], cmap=green_cmap, alpha=0.35)
    ax.set_title(f"{roi_name} overlay on slice {slice_index}")
    ax.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
