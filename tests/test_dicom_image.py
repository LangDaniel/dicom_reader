from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import ExplicitVRLittleEndian

import dicom_reader


def _write_dicom_slice(path, z, uid, rows=4, cols=4, values=None):
    if values is None:
        values = np.arange(rows * cols, dtype=np.uint16).reshape(rows, cols)

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.PixelSpacing = [1.0, 1.0]
    ds.ImagePositionPatient = [0.0, 0.0, z]
    ds.SOPInstanceUID = uid
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    ds.PixelData = values.astype(np.uint16).tobytes()
    ds.is_implicit_VR = False
    ds.is_little_endian = True
    ds.save_as(str(path), write_like_original=False)


def test_dicom_image_smoke(tmp_path):
    slice_dir = tmp_path / 'slices'
    slice_dir.mkdir()

    first_path = slice_dir / 'slice_1.dcm'
    second_path = slice_dir / 'slice_2.dcm'
    _write_dicom_slice(first_path, z=0.0, uid='uid-1')
    _write_dicom_slice(second_path, z=10.0, uid='uid-2')

    img = dicom_reader.DICOMImage(str(slice_dir))

    assert len(img) == 2
    assert img.get_shape() == (4, 4, 2)
    np.testing.assert_allclose(img.get_spacing(), [1.0, 1.0, 10.0])
    assert img.get_origin() == [0.0, 0.0, 0.0]


def test_dicom_image_pixel_array(tmp_path):
    slice_dir = tmp_path / 'slices'
    slice_dir.mkdir()

    first_path = slice_dir / 'slice_1.dcm'
    second_path = slice_dir / 'slice_2.dcm'
    first_values = np.ones((4, 4), dtype=np.uint16) * 10
    second_values = np.ones((4, 4), dtype=np.uint16) * 20
    _write_dicom_slice(first_path, z=0.0, uid='uid-1', values=first_values)
    _write_dicom_slice(second_path, z=10.0, uid='uid-2', values=second_values)

    img = dicom_reader.DICOMImage(str(slice_dir))
    data = img.get_pixel_array()

    assert data.shape == (4, 4, 2)
    assert data[:, :, 0].sum() == 160
    assert data[:, :, 1].sum() == 320
