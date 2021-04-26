import numpy as np

class Preprocessing():
    
    def __init__(
        self,
        scale_linear=False,
        ROI=False,
        target_shape=False,
        spacing=False,
        order=1,
        **kwargs):
        
        if isinstance(scale_linear, bool):
            self.rescale = False
        else:
            self.scale_linear = scale_linear
            self.rescale = True
        
        if isinstance(ROI, bool):
            self.use_roi = False
        else:
            self.ROI = np.array(ROI)
            self.use_roi = True

        if isinstance(target_shape, bool):
            self.use_target_shape = False
        else:
            self.target_shape = np.array(target_shape)
            self.use_target_shape = True
        
        if isinstance(spacing, bool):
            self.change_spacing = False
        else:
            self.spacing = np.array(spacing)
            self.change_spacing = True
            
        self.order = order
    
    
    def crop_data(self, data, crop_region):
        shape = data.shape
        
        # if the ROI size is 3 it marks the center
        if crop_region.size == 3:
            init = (crop_region - self.target_shape // 2)
            
            # check that the min idx is => 0
            init = np.clip(init, a_min=0, a_max=None)
            
            # check that the max idx is <= axis size
            for idx in range(0, len(init)):
                if (init[idx] + self.target_shape[idx]) > shape[idx]:
                    init[idx] = shape[idx] - self.target_shape[idx]
            
            ii = init
            ff = init + self.target_shape
        
        # if the ROI size is 6 it marks a bounding box
        elif crop_region.size == 6:
            if self.use_target_shape:
                for idx in range(0, len(crop_region) // 2):
                    
                    # check if the the target shape is bigger than the ROI
                    ROI_dist = crop_region[2*idx + 1] - crop_region[2*idx]
                    missing = self.target_shape[idx] - ROI_dist
                    
                    # if so add the missing part
                    if (missing > 0):
                        
                        # check that the min idx is => 0
                        crop_region[2*idx] = max(0, crop_region[2*idx] - missing // 2)
                        crop_region[2*idx + 1] = crop_region[2*idx] + self.target_shape[idx]
                        
                        # check that the max idx is <= axis size
                        if crop_region[2*idx + 1] > shape[idx]:
                            crop_region[2*idx] = shape[idx] - self.target_shape[idx]
                            crop_region[2*idx + 1] = crop_region[2*idx] + self.target_shape[idx] 
          
            ii = crop_region[::2]
            ff = crop_region[1::2]

        data = data[ii[0]:ff[0], ii[1]:ff[1], ii[2]:ff[2]]
        
        # use zero padding if the image is to small
        if self.use_target_shape:            
            if (shape < self.target_shape).any():
                data_ = np.zeros(self.target_shape)
                data_[:shape[0], :shape[1], shape[2]] = data
                data = data_

        return data
    
    def linear_rescale(self, data):
        xlow = self.scale_linear[0][0]
        xhigh = self.scale_linear[0][1]
        ylow = self.scale_linear[1][0]
        yhigh = self.scale_linear[1][1]
        
        m = (yhigh - ylow) / (xhigh - xlow)
        b = ylow - m * xlow
        
        out = m * data + b
        out = np.clip(out, ylow, yhigh)
        
        return out
    
    def resample(self, data, initial_spacing):
        shape = data.shape
        
        resize_factor = np.round(shape * (initial_spacing / self.spacing)) / shape
        
        return zoom(data, zoom=resize_factor, order=self.order), resize_factor
    
    def preprocess(self, dataset, crop_region=False):
        data = dataset.get_pixel_array()

        if isinstance(crop_region, bool):
            crop = False
        else:
            crop_region = np.array(crop_region)
            crop = True

        if self.change_spacing:
            spacing = dataset.get_spacing()
            data, zoom_fac = self.resample(data, spacing)
        
            if crop:
                if (crop_region.size == 3) or (crop_region.size == 2):
                    crop_region = zoom_fac * crop_region
                elif (crop_region.size == 6) or (crop_region.size == 4):
                    crop_region[::2] = zoom_fac * crop_region[::2]
                    crop_region[1::2] = zoom_fac * crop_region[1::2]
    
        if crop:
            data = self.crop_data(data, crop_region)
        
        if self.rescale:
            data = self.linear_rescale(data)
            
        return data
