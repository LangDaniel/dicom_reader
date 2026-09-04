from pathlib import Path

import cv2
import numpy as np
import pydicom as pydcm


class DICOMImage:
    """Represent a 3D DICOM image volume stored as axial slices."""

    def __init__(self, dcm_folder):
        """Load a folder of DICOM slices and sort them by patient position."""
        slice_paths = [
            str(ff) for ff in Path(dcm_folder).iterdir() if ff.suffix == '.dcm'
        ]

        self.instance_uids, sorted_idx = self.sort_instance_uids(slice_paths)
        self.slice_paths = np.array(slice_paths)[sorted_idx]

    def __len__(self):
        return len(self.slice_paths)

    def sort_instance_uids(self, slice_paths):
        """Return slice instance UIDs sorted by height."""
        heights = np.empty(len(slice_paths))
        uids = np.empty(len(slice_paths), dtype='U100')

        for idx, slz in enumerate(slice_paths):
            ds = pydcm.read_file(slz)
            heights[idx] = ds.ImagePositionPatient[-1]
            uids[idx] = ds.SOPInstanceUID

        sorted_idx = np.argsort(heights)
        return uids[sorted_idx[::-1]], sorted_idx

    def get_shape(self):
        """Return the image size as (rows, columns, num_slices)."""
        ds = pydcm.read_file(self.slice_paths[0])
        return (ds.Rows, ds.Columns, len(self.slice_paths))

    def get_orientation(self, precision=0.005):
        """Return the image orientation and validate it across slices."""
        orient = pydcm.read_file(self.slice_paths[0]).ImageOrientationPatient

        for idx in range(1, len(self.slice_paths)):
            ds = pydcm.read_file(self.slice_paths[idx])
            if ds.ImageOrientationPatient != orient:
                raise ValueError('orientation changed')

        orient = np.array(orient)
        orient = np.array([*orient[abs(orient) > precision], 1])
        return orient

    def get_origin(self):
        """Return the patient position of the first slice."""
        ds = pydcm.read_file(self.slice_paths[0])
        return ds.ImagePositionPatient

    def get_slice_positions(self):
        """Return the patient position for every slice."""
        positions = np.empty((3, len(self.slice_paths)))
        for idx, slize in enumerate(self.slice_paths):
            ds = pydcm.read_file(slize)
            positions[:, idx] = ds.ImagePositionPatient
        return positions

    def get_pos_from_uid(self, uid):
        """Return the patient position for a given SOP Instance UID."""
        for slize in self.slice_paths:
            ds = pydcm.read_file(slize)
            if ds.SOPInstanceUID == uid:
                return ds.ImagePositionPatient
        return False

    def get_spacing(self, precision=None):
        """Return the image spacing in x, y, and z directions."""
        if precision is None:
            precision = [0.05, 0.05, 0.05]

        pix_spacing = np.empty((2, len(self.slice_paths)))
        heights = np.empty(len(self.slice_paths))
        for idx, slize in enumerate(self.slice_paths):
            ds = pydcm.read_file(slize)
            pix_spacing[:, idx] = ds.PixelSpacing
            heights[idx] = float(ds.ImagePositionPatient[-1])

        slice_spacing = np.diff(heights)

        if not (
            (pix_spacing[:, 0] - pix_spacing[0, 0]) < precision[0]
        ).all():
            raise ValueError('x spacing changed')
        if not (
            (pix_spacing[:, 1] - pix_spacing[0, 1]) < precision[1]
        ).all():
            raise ValueError('y spacing changed')
        if not (
            (slice_spacing - slice_spacing[0]) < precision[-1]
        ).all():
            raise ValueError('z spacing changed')

        return np.array([
            pix_spacing[0, 0],
            pix_spacing[0, 1],
            slice_spacing[0],
        ])

    def get_pixel_array(self):
        """Read all image slices and return a 3D NumPy array."""
        ct_class_uid = '1.2.840.10008.5.1.4.1.1.2'
        pet_class_uid = '1.2.840.10008.5.1.4.1.1.128'
        mri_class_uid = '1.2.840.10008.5.1.4.1.1.4'

        img = np.empty(self.get_shape())

        for idx, slz in enumerate(self.slice_paths):
            ds = pydcm.read_file(slz)
            data = ds.pixel_array
            class_uid = ds.SOPClassUID
            if (class_uid == ct_class_uid) or (class_uid == pet_class_uid):
                data = (
                    (float(ds.RescaleSlope) * data)
                    + float(ds.RescaleIntercept)
                )
            elif class_uid == mri_class_uid:
                pass
            else:
                raise ValueError('image format currently not able to read yet')
            img[:, :, idx] = data
        return img

    def get_manufacturer(self):
        """Return the manufacturer name of slice 0."""
        ds = pydcm.read_file(self.slice_paths[0])
        try:
            manu = ds.Manufacturer
        except Exception:
            manu = 'None'
        return manu

    def get_model(self):
        """Return the manufacturer model of slice 0."""
        ds = pydcm.read_file(self.slice_paths[0])
        try:
            model = ds.ManufacturerModelName
        except Exception:
            model = 'None'
        return model

    def get_tag(self, tag, slz=None):
        """Return a specific DICOM tag for a given slice."""
        if slz is None:
            slz = self.__len__() // 2
        ds = pydcm.read_file(self.slice_paths[slz])
        try:
            value = getattr(ds, tag)
        except Exception:
            value = 'None'
        return value

    def list_tags(self, slz=None):
        """Return all tags for a given slice."""
        if slz is None:
            slz = self.__len__() // 2
        ds = pydcm.read_file(self.slice_paths[slz])
        return dir(ds)


class DICOMContour:
    """Abstract base class for DICOM contour-like objects."""

    def __init__(self, file_path, origin, spacing, shape):
        self.ds = pydcm.read_file(str(file_path))
        self.origin = origin
        self.spacing = spacing
        self.shape = shape

    def get_shape(self):
        return self.shape

    def get_spacing(self):
        return self.spacing

    def get_origin(self):
        return self.origin

    @property
    def get_pixel_array(self):
        raise NotImplementedError('Subclasses should implement this!')

    @property
    def get_contour(self):
        raise NotImplementedError('Subclasses should implement this!')

    def get_bbox(self, order='numpy'):
        """Return a global bounding box of the contour."""
        order = order.lower()
        assert order in [
            'numpy', 'cv2', 'row_first', 'col_first'
        ], 'order not found'

        contour = self.get_contour()
        bbox = np.zeros(6)
        bbox[::2] = np.min(contour, axis=0)
        bbox[1::2] = np.max(contour, axis=0)
        bbox = bbox.astype(int)

        if (order == 'cv2') or (order == 'col_first'):
            return bbox

        bbox[:4] = bbox[2], bbox[3], bbox[0], bbox[1]
        return bbox

    def get_center(self, order='numpy'):
        """Return the center of the contour region."""
        order = order.lower()
        assert order in [
            'numpy', 'cv2', 'row_first', 'col_first'
        ], 'order not found'

        contour = self.get_contour()
        center = np.mean(contour, axis=0).astype(int)

        if (order == 'cv2') or (order == 'col_first'):
            return center

        center[0], center[1] = center[1], center[0]
        return center


class DICOMSeg(DICOMContour):
    """Read a DICOM SEG file and convert it to a 3D mask."""

    def __init__(self, file_path, origin, spacing, shape):
        super().__init__(
            file_path=file_path,
            origin=origin,
            spacing=spacing,
            shape=shape,
        )

        self.seg_origin = (
            self.ds.PerFrameFunctionalGroupsSequence[0]
            .PlanePositionSequence[0]
            .ImagePositionPatient
        )

        self.check_axial_origin()

        self.seg_shape = np.array((
            self.ds.Rows,
            self.ds.Columns,
            self.ds.NumberOfFrames,
        )).astype(int)

        self.pad_height = False
        self.check_shape()

    def check_axial_origin(self, precision=None):
        """Validate the SEG origin against the image origin."""
        if precision is None:
            precision = [0.05, 0.05]

        for idx in range(0, 2):
            if abs(self.seg_origin[idx] - self.origin[idx]) > precision[idx]:
                print(f'{self.seg_origin[idx]} vs {self.origin[idx]}')
                raise ValueError('axial origin differs significantly')

    def check_shape(self):
        """Validate SEG shape against the reference image shape."""
        for idx in range(0, 2):
            if self.shape[idx] != self.seg_shape[idx]:
                raise ValueError('axial shape differs')
        if self.shape[2] > self.seg_shape[2]:
            self.pad_height = True

    def padding(self, data):
        """Pad the segmentation volume to match the reference image."""
        dist = self.seg_origin[2] - self.origin[2]
        pix_dist = dist / self.spacing[2]
        if (pix_dist).is_integer():
            pix_dist = int(pix_dist)
        else:
            raise ValueError('non integer pixel distance')

        data_ = np.zeros(self.shape)
        data_[:, :, pix_dist:pix_dist + self.seg_shape[2]] = data
        return data_

    def get_pixel_array(self):
        """Return the SEG data as a 3D NumPy array."""
        data = self.ds.pixel_array
        data = np.moveaxis(data, 0, -1)

        if self.pad_height:
            data = self.padding(data)
        return data

    def get_contour(self):
        """Convert active segmentation voxels to contour coordinates."""
        contour = np.where(self.get_pixel_array())
        con = np.empty((len(contour[0]), 3))
        con[:, 0] = contour[1]
        con[:, 1] = contour[0]
        con[:, 2] = contour[2]
        return con.astype(int)


class DICOMStruct(DICOMContour):
    """Read an RTSTRUCT file and convert a selected ROI to a mask."""

    def __init__(self, file_path, origin, spacing, shape, ROI=False):
        super().__init__(
            file_path=file_path,
            origin=origin,
            spacing=spacing,
            shape=shape,
        )

        if isinstance(ROI, str):
            self.ROI_idx = self.get_ROI_index(ROI)
        else:
            self.ROI_idx = ROI

    def set_ROI_idx(self, ROI_name):
        """Set the active ROI by name."""
        self.ROI_idx = self.get_ROI_index(ROI_name)

    def get_ROI_names(self):
        """Return all defined ROI names."""
        names = []
        for item in self.ds.StructureSetROISequence:
            names.append(item.ROIName)
        return names

    def get_ROI_index(self, ROI_name):
        """Return the index of an ROI name."""
        for ii, nn in enumerate(self.get_ROI_names()):
            if nn == ROI_name:
                return ii
        print('ROI name not found use one of')
        print(self.get_ROI_names())
        return False

    def coordinates_to_pixel(self, coordinates):
        """Convert scanner coordinates to voxel indices."""
        x_idx = abs(self.origin[0] - coordinates[0]) / self.spacing[0]
        y_idx = abs(self.origin[1] - coordinates[1]) / self.spacing[1]
        z_idx = abs(self.origin[2] - coordinates[2]) / self.spacing[2]

        return np.array([x_idx, y_idx, z_idx]).astype(int)

    def get_contour(self, mode='pixel'):
        """Return all contour points for the active ROIs."""
        assert mode in ['pixel', 'coordinates'], 'mode not found'

        if isinstance(self.ROI_idx, bool):
            raise ValueError('not ROI is set: use set_ROI_idx to do so')

        roi_seq = self.ds.ROIContourSequence[self.ROI_idx]
        coordinates = []
        for contour in roi_seq.ContourSequence:
            coordinates = coordinates + list(contour.ContourData)

        coordinates = np.array(coordinates).astype(float)
        if mode == 'coordinates':
            return coordinates

        for ii in np.arange(0, len(coordinates), 3):
            coordinates[ii:ii + 3] = self.coordinates_to_pixel(
                coordinates[ii:ii + 3]
            )
        return coordinates.astype(int).reshape(-1, 3)

    def get_pixel_array(self):
        """Return the selected contour as a 3D binary mask."""
        if isinstance(self.ROI_idx, bool):
            raise ValueError('not ROI is set: use set_ROI_idx to do so')

        data = np.zeros(self.shape)
        contour = self.get_contour(mode='pixel')
        for z_idx in range(0, self.shape[-1]):
            slice_con = np.array(contour[contour[:, -1] == z_idx][:, :2])
            if slice_con.size:
                data[:, :, z_idx] = cv2.drawContours(
                    image=np.array(data[:, :, z_idx]),
                    contours=[slice_con],
                    contourIdx=-1,
                    color=(1, 1, 1),
                    thickness=-1,
                )
        return data
