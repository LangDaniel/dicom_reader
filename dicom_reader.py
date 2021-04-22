from pathlib import Path
import cv2
import numpy as np
import pydicom as pydcm

class DICOMImage():
    def __init__(self, dcm_folder):
        slice_paths = [str(ff) for ff in Path(dcm_folder).iterdir() if ff.suffix == '.dcm']

        self.instance_uids, sorted_idx = self.sort_instance_uids(slice_paths)
        self.slice_paths = np.array(slice_paths)[sorted_idx]

    def sort_instance_uids(self, slice_paths):
        '''returns the slice instance uid sorted by height'''
        heights = np.empty(len(slice_paths))
        uids = np.empty(len(slice_paths), dtype='U100')

        for idx, slz in enumerate(slice_paths):
            ds = pydcm.read_file(slz)
            heights[idx] = ds.ImagePositionPatient[-1]
            uids[idx] = ds.SOPInstanceUID
        sorted_idx = heights.argsort()
        return uids[sorted_idx[::-1]], sorted_idx

    def get_size(self):
        '''get the size of the image'''
        ds = pydcm.read_file(self.slice_paths[0])
        return (ds.Rows, ds.Columns, len(self.slice_paths))

    def get_origin(self):
        '''get patient position of first (lowest) slice'''
        ds = pydcm.read_file(self.slice_paths[0])
        return ds.ImagePositionPatient

    def get_slice_positions(self):
        ''' get the patient position of all slice_paths '''
        positions = np.empty((3, len(self.slice_paths)))
        for idx, slize in enumerate(self.slice_paths):
            ds = pydcm.read_file(slize)
            positions[:, idx] = ds.ImagePositionPatient
        return positions

    def get_pos_from_uid(self, uid):
        ''' takes an instance uid as argument and returns the corresponding
            patient position of the slice
        '''
        for slize in self.slice_paths:
            ds = pydcm.read_file(slize)
            if ds.SOPInstanceUID == uid:
                return ds.ImagePositionPatient
        return False

    def get_spacing(self, precision=[0.05, 0.05, 0.05]):
        '''get the image level spacing'''
        pix_spacing = np.empty((2, len(self.slice_paths)))
        heights = np.empty(len(self.slice_paths))
        for idx, slize in enumerate(self.slice_paths):
            ds = pydcm.read_file(slize)
            pix_spacing[:, idx] = ds.PixelSpacing
            heights[idx] = float(ds.SliceLocation)

        slice_spacing = np.diff(heights)

        # check if the spacing is the same for the complete image
        if not ((pix_spacing[:, 0] - pix_spacing[0, 0]) < precision[0]).all():
            raise ValueError('x spacing changed')
        if not ((pix_spacing[:, 1] - pix_spacing[0, 1]) < precision[1]).all():
            raise ValueError('y spacing changed')
        if not ((slice_spacing - slice_spacing[0]) < precision[-1]).all():
            raise ValueError('z spacing changed')

        return np.array([pix_spacing[0, 0], pix_spacing[0, 1], slice_spacing[0]])

    def get_pixel_array(self):
        '''read all the images and returns them sorted by their height'''
        img = np.empty(self.get_size())

        for idx, slz in enumerate(self.slice_paths):
            ds = pydcm.read_file(slz)
            img[:, :, idx] = ds.pixel_array
        return img

class DICOMStruct():
    def __init__(self, file_path, origin, spacing, shape):
        self.ds = pydcm.read_file(str(file_path))
        self.origin = origin
        self.spacing = spacing
        self.shape = shape

    def get_ROI_names(self):
        '''returns names off all ROI'''
        names = []
        for item in self.ds.StructureSetROISequence:
            names.append(item.ROIName)
        return names

    def get_ROI_index(self, ROI_name):
        '''returns the index for a ROI name'''
        for ii, nn in enumerate(self.get_ROI_names()):
            if nn == ROI_name:
                return ii
        return False

    def coordinates_to_pixel(self, coordiantes):
        ''' takes a scanner coordiante (x, y, z) and a refenerce CT and returns
            the ijk pixel data coordiate '''
        x_idx = abs(self.origin[0] - coordiantes[0]) / self.spacing[0]
        y_idx = abs(self.origin[1] - coordiantes[1]) / self.spacing[1]
        z_idx = abs(self.origin[2] - coordiantes[2]) / self.spacing[2]

        return np.array([x_idx, y_idx, z_idx]).astype(int)

    def get_contour_data(self, ROI, mode='pixel'):
        '''get the complete contour data for a given region of interesst'''
        assert (mode == 'pixel') or (mode == 'coordinates'), 'mode not found'

        if isinstance(ROI, str):
            roi_idx = self.get_ROI_index(ROI)
        else:
            roi_idx = ROI

        roi_seq = self.ds.ROIContourSequence[roi_idx]
        coordinates = []
        for contour in roi_seq.ContourSequence:
            coordinates = coordinates + list(contour.ContourData)

        coordinates = np.array(coordinates).astype(float)
        if mode == 'coordinates':
            return coordinates

        for ii in np.arange(0, len(coordinates), 3):
            coordinates[ii: ii+3] = self.coordinates_to_pixel(coordinates[ii: ii+3])
        return coordinates.astype(int)

    def get_pixel_array(self, ROI):
        '''returns the contour as numpy array'''
        if isinstance(ROI, str):
            roi_idx = self.get_ROI_index(ROI)
        else:
            roi_idx = ROI

        data = np.zeros(self.shape)
        contour = self.get_contour_data(roi_idx, mode='pixel')
        contour = contour.reshape(-1, 3)
        for z_idx in range(0, self.shape[-1]):
            slice_con = np.array(contour[contour[:, -1] == z_idx][:, :2])
            if slice_con.size:
                data[:, :, z_idx] = cv2.drawContours(
                    image=np.array(data[:, :, z_idx]),
                    contours=[slice_con],
                    contourIdx=-1,
                    color=(1, 1, 1),
                    thickness=-1
                )
        return data

    def get_bbox(self, ROI):
        '''returns a global bounding box for the contour'''
        if isinstance(ROI, str):
            roi_idx = self.get_ROI_index(ROI)
        else:
            roi_idx = ROI

        contour = self.get_contour_data(roi_idx).reshape(-1, 3)
        bbox = np.zeros(6)
        bbox[::2] = np.min(contour, axis=0)
        bbox[1::2] = np.max(contour, axis=0)

        return bbox.astype(int)
