from pathlib import Path
import cv2
import numpy as np
import pydicom as pydcm

class DICOMImage():
    ''' dicom image class

        Args:
            dcm_folder: str, path to the folder of the dicom slices
    '''

    def __init__(self, dcm_folder):
        slice_paths = [str(ff) for ff in Path(dcm_folder).iterdir() if ff.suffix == '.dcm']

        self.instance_uids, sorted_idx = self.sort_instance_uids(slice_paths)
        self.slice_paths = np.array(slice_paths)[sorted_idx]

    def __len__(self):
        return len(self.slice_paths)

    def sort_instance_uids(self, slice_paths):
        '''returns the slice instance uids sorted by height'''

        heights = np.empty(len(slice_paths))
        uids = np.empty(len(slice_paths), dtype='U100')

        for idx, slz in enumerate(slice_paths):
            ds = pydcm.read_file(slz)
            heights[idx] = ds.ImagePositionPatient[-1]
            uids[idx] = ds.SOPInstanceUID
        sorted_idx = heights.argsort()
        return uids[sorted_idx[::-1]], sorted_idx

    def get_shape(self):
        '''get the size of the image'''

        ds = pydcm.read_file(self.slice_paths[0])
        return (ds.Rows, ds.Columns, len(self.slice_paths))

    def get_orientation(self, precision=0.005):
        '''get the orientation of the image'''

        # get orientation of first slice
        orient = pydcm.read_file(self.slice_paths[0]).ImageOrientationPatient

        # check that all other slices have the same orientation
        for ii in range(1, len(self.slice_paths)):
            ds = pydcm.read_file(self.slice_paths[ii])
            if ds.ImageOrientationPatient != orient:
                raise ValueError('orientation changed')

        orient = np.array(orient)
        orient = np.array([*orient[abs(orient) > precision], 1])

        return orient

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
        '''get the image level spacing

        Args:
            precision: list with a len of image dimensions
                specifies how much the spacing between slices can vary
                befor throwing an error
        '''

        pix_spacing = np.empty((2, len(self.slice_paths)))
        heights = np.empty(len(self.slice_paths))
        for idx, slize in enumerate(self.slice_paths):
            ds = pydcm.read_file(slize)
            pix_spacing[:, idx] = ds.PixelSpacing
            heights[idx] = float(ds.ImagePositionPatient[-1])
            #heights[idx] = float(ds.SliceLocation)

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
        CT_class_uid = '1.2.840.10008.5.1.4.1.1.2'
        PET_class_uid = '1.2.840.10008.5.1.4.1.1.128'
        MRI_class_uid = '1.2.840.10008.5.1.4.1.1.4'

        img = np.empty(self.get_shape())

        for idx, slz in enumerate(self.slice_paths):
            ds = pydcm.read_file(slz)
            data = ds.pixel_array
            class_uid = ds.SOPClassUID
            if (class_uid == CT_class_uid) or (class_uid == PET_class_uid):
                data = (float(ds.RescaleSlope) * data) + float(ds.RescaleIntercept)
            elif class_uid == MRI_class_uid:
                pass
                #print('MRI image: no rescaling (temporarely !!!!!!!!!!!!')
            else:
                raise ValueError('image format currently not able to read yet')
            img[:, :, idx] = data
        return img

    def get_manufacturer(self):
        '''returns the manufacturer of slice 0 (no check for other slices)'''
        ds = pydcm.read_file(self.slice_paths[0])
        try:
            manu = ds.Manufacturer
        except:
            manu = 'None'
        return manu

    def get_model(self):
        ''' returns the manufacturer model name of slice 0
            (no check for other slices)
        '''
        ds = pydcm.read_file(self.slice_paths[0])
        try:
            model = ds.ManufacturerModelName
        except:
            model = 'None'
        return model

    def get_tag(self, tag, slz=None):
        ''' returns a specific tag for slice "slz"
            (no check for other slices)
        '''
        if slz == None:
            slz = self.__len__()//2
        ds = pydcm.read_file(self.slice_paths[slz])
        try:
            value = getattr(ds, tag)
        except:
            value = 'None'
        return value

    def list_tags(self, slz=None):
        ''' returns all tags of slice "slz"
        '''
        if slz == None:
            slz = self.__len__()//2
        ds = pydcm.read_file(self.slice_paths[slz])
        return dir(ds)

class DICOMContour():
    ''' abstract dicom contour class, parent to DICOMStruct and DICOMSeg

        Args:
            file_path: str, path to the dicom file
            origin: list, patient origin of the lowest slice
            spacing: list, pixel spacing of the corresponding image file
    '''

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
        raise NotImplementedError("Subclasses should implement this!")

    @property
    def get_contour(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_bbox(self, order='numpy'):
        '''returns a global bounding box for the contour

        Args:
            order: one of "numpy", "cv2", "row_first" or "col_first" specifies
                the order of the axis. (numpy == row_first, cv2 == col_first)
        '''
        #if isinstance(self.ROI_idx, bool):
        #    raise ValueError('not ROI is set: use set_ROI_idx to do so')

        order = order.lower()
        assert order in ['numpy', 'cv2', 'row_first', 'col_first'], 'order not found'

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
        ''' returns the center of the region of interesst

        Args:
            order: one of "numpy", "cv2", "row_first" or "col_first" specifies
                the order of the axis. (numpy == row_first, cv2 == col_first)
        '''

        order = order.lower()
        assert order in ['numpy', 'cv2', 'row_first', 'col_first'], 'order not found'

        contour = self.get_contour()
        center = np.mean(contour, axis=0).astype(int)

        if (order == 'cv2') or (order == 'col_first'):
            return center

        center[0], center[1] = center[1], center[0]
        return center


class DICOMSeg(DICOMContour):
    ''' dicom seg class, lets you read SEG files

        Args:
            file_path: str, path to the RTSTRUCT dicom file
            origin: list, patient origin of the lowest slice
            spacing: list, pixel spacing of the corresponding image file
    '''

    def __init__(self, file_path, origin, spacing, shape):
        super(DICOMSeg, self).__init__(
            file_path=file_path,
            origin=origin,
            spacing=spacing,
            shape=shape,
        )

        self.seg_origin = (self.ds
                            .PerFrameFunctionalGroupsSequence[0]
                                .PlanePositionSequence[0]
                                    .ImagePositionPatient)

        self.check_axial_origin()

        self.seg_shape = np.array((
            self.ds.Rows,
            self.ds.Columns,
            self.ds.NumberOfFrames
        )).astype(int)

        self.pad_height = False
        self.check_shape()

    def check_axial_origin(self, precision=[0.05, 0.05]):
        for idx in range(0, 2):
            if abs(self.seg_origin[idx] - self.origin[idx]) > precision[idx]:
                print(f'{self.seg_origin[idx]} vs {self.origin[idx]}')
                raise ValueError('axial origin differs significantly')

    def check_shape(self):
        for idx in range(0, 2):
            if self.shape[idx] != self.seg_shape[idx]:
                raise ValueError('axial shape differs')
        if self.shape[2] > self.seg_shape[2]:
            self.pad_height = True

    def padding(self, data):
        dist = self.seg_origin[2] - self.origin[2]
        pix_dist = dist / self.spacing[2]
        if (pix_dist).is_integer():
            pix_dist = int(pix_dist)
        else:
            raise ValueError('non integer pixel distance')

        data_ = np.zeros(self.shape)
        data_[:, :, pix_dist:pix_dist+self.seg_shape[2]] = data

        return data_

    def get_pixel_array(self):
        data = self.ds.pixel_array
        data = np.moveaxis(data, 0, -1)

        if self.pad_height:
            data = self.padding(data)
        return data

    def get_contour(self):
        contour = np.where(self.get_pixel_array())
        con = np.empty((len(contour[0]), 3))
        # reshape and change form numpy to cv2 order
        con[:, 0] = contour[1]
        con[:, 1] = contour[0]
        con[:, 2] = contour[2]
        return con.astype(int)


class DICOMStruct(DICOMContour):
    ''' dicom struct class, lets you read RTSTRUCT files and convert them
        to pixel data

        Args:
            file_path: str, path to the RTSTRUCT dicom file
            origin: list, patient origin of the lowest slice
            spacing: list, pixel spacing of the corresponding image file
            shape: list, shape of the corresponding image file
            ROI: str or int, specification of the name/index of the ROI to be handled
    '''

    def __init__(self, file_path, origin, spacing, shape, ROI=False):
        super(DICOMStruct, self).__init__(
            file_path=file_path,
            origin=origin,
            spacing=spacing,
            shape=shape
        )

        if isinstance(ROI, str):
            self.ROI_idx = self.get_ROI_index(ROI)
        else:
            self.ROI_idx = ROI

    def set_ROI_idx(self, ROI_name):
        self.ROI_idx = self.get_ROI_index(ROI_name)

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
        print('ROI name not found use one of')
        print(self.get_ROI_names())
        return False

    def coordinates_to_pixel(self, coordiantes):
        ''' takes a scanner coordiante (x, y, z) and returns
            the ijk pixel data coordiate
        '''

        x_idx = abs(self.origin[0] - coordiantes[0]) / self.spacing[0]
        y_idx = abs(self.origin[1] - coordiantes[1]) / self.spacing[1]
        z_idx = abs(self.origin[2] - coordiantes[2]) / self.spacing[2]

        return np.array([x_idx, y_idx, z_idx]).astype(int)

    def get_contour(self, mode='pixel'):
        ''' get the complete contour data for a given region of interesst
            specified by self.ROI

        Args:
            mode: "pixel" or "coordinates",
                for pixel indices are returned
                for coordiante distances in mm are returned
        '''

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
            coordinates[ii: ii+3] = self.coordinates_to_pixel(coordinates[ii: ii+3])
        return coordinates.astype(int).reshape(-1, 3)

    def get_pixel_array(self):
        '''returns the contour as numpy array

        Args:
            None
        '''
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
                    thickness=-1
                )
        return data
