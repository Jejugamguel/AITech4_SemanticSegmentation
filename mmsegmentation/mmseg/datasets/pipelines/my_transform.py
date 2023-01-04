import glob
import numpy as np
import cv2
import albumentations as A
from numpy import random
from ..builder import PIPELINES



from PIL import Image


@PIPELINES.register_module()
class ChannelShuffle(object):
    """
    Randomly rearrange channels of the input RGB image.
    
    input : dict
    output : dict
    
    Args:
        prob (float, optional) : probability of applying ChannelShuffle
        
    """
    def __init__(self,prob):
        self.prob = prob

        if prob is not None:
            assert prob >= 0 and prob <= 1

    def __call__(self, results):
        if 'channelshuffle' not in results:
            channelshuffle = True if np.random.rand() < self.prob else False
            results['channelshuffle'] = channelshuffle

        if results['channelshuffle']:
            image = results['img']
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = results['gt_semantic_seg']

            transform = A.Compose([
                        A.ChannelShuffle(p=self.prob),
                        ],
                        p=1)
            transformed = transform(image=image, mask=mask )
            results['img'] = transformed['image']
            results['gt_semantic_seg'] = transformed['mask']
        return results

    
    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'



@PIPELINES.register_module()
class CLAHE2(object):
    """
    Apply Contrast Limited Adaptive Histogram Equalization to the input image.
    
    input : dict
    output : dict
    
    Args:
        prob (float, optional) : probability of applying ChannelShuffle
        clip_limit (float or [float, float]) :upper threshold value for contrast limiting. If clip_limit is a single float value, the range will be (1, clip_limit). Default: (1, 4).
        tile_grid_size ([int, int]) : size of grid for histogram equalization. Default: (8, 8).
    """
    def __init__(self,prob,clip_limit=4.0, tile_grid_size=(8, 8)):
        self.prob = prob
        self.clip_limit= clip_limit
        self.tile_grid_size = tile_grid_size
        if prob is not None:
            assert prob >= 0 and prob <= 1



    def __call__(self, results):
        if 'clahe' not in results:
            clahe = True if np.random.rand() < self.prob else False
            results['clahe'] = clahe

        if results['clahe']:
            image = results['img']
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = results['gt_semantic_seg']

            transform = A.Compose([
                        A.CLAHE(clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size,p=self.prob),
                        ],
                        p=1)
            transformed = transform(image=image, mask=mask)
            results['img'] = transformed['image']
            results['gt_semantic_seg'] = transformed['mask']
        return results

    
    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class Transpose2(object):
    """
    Transpose the input by swapping rows and columns.
    
    input : dict
    output : dict
    
    Args:
        prob (float, optional) : probability of applying ChannelShuffle
        
    """
    def __init__(self,prob):
        self.prob = prob

        if prob is not None:
            assert prob >= 0 and prob <= 1




    def __call__(self, results):
        if 'transpose' not in results:
            transpose = True if np.random.rand() < self.prob else False
            results['transpose'] = transpose

        if results['transpose']:
            image = results['img']
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = results['gt_semantic_seg']

            transform = A.Compose([
                        A.Transpose(p=self.prob),
                        ],
                        p=1)
            transformed = transform(image=image, mask=mask )
            results['img'] = transformed['image']
            results['gt_semantic_seg'] = transformed['mask']
        return results

    
    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'

@PIPELINES.register_module()
class CropNonEmptyMaskIfExists(object):
    """
    Transpose the input by swapping rows and columns.
    
    input : dict
    output : dict
    
    Args:
        prob (float, optional) : probability of applying ChannelShuffle
        
    """
    def __init__(self,height=256,width=256,prob=0.5):
        self.prob = prob
        self.height = height
        self.width = width

        if prob is not None:
            assert prob >= 0 and prob <= 1




    def __call__(self, results):
        if 'transpose' not in results:
            transpose = True if np.random.rand() < self.prob else False
            results['transpose'] = transpose

        if results['transpose']:
            image = results['img']
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = results['gt_semantic_seg']

            transform = A.Compose([
                        A.CropNonEmptyMaskIfExists(height=self.height, width=self.width, p=self.prob),
                        ],
                        p=1)
            transformed = transform(image=image, mask=mask )
            results['img'] = transformed['image']
            results['gt_semantic_seg'] = transformed['mask']
        return results

    
    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'



@PIPELINES.register_module()
class RandomCutmix(object):
    """
    Cutout random patch & Paste same positional patch of random image
    
    input : dict
    output : dict
    
    Args:
        prob (float, optional) : probability of applying Cutmix
        box_scale (tuple, optional) : size of cutmix patch
        
    Advanced (Yet):
        specify cutmix patch range (inner box)
    """
    def __init__(self, prob=None, patch_scale=(128, 128)):
        self.prob = prob
        self.patch_scale = patch_scale
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert patch_scale[0] <= 512 and patch_scale[1] <= 512
        assert isinstance(patch_scale[0], int) and isinstance(patch_scale[1], int)
    
    def make_cutmix(self, img, patch_img, mask, patch_mask, tf_row_max, tf_col_max):
        tf_row = random.randint(0, tf_row_max)
        tf_col = random.randint(0, tf_col_max)
        img[tf_row:tf_row+self.patch_scale[0], tf_col:tf_col+self.patch_scale[1], :] \
            = patch_img[tf_row:tf_row+self.patch_scale[0], tf_col:tf_col+self.patch_scale[1], :]
        
        mask[tf_row:tf_row+self.patch_scale[0], tf_col:tf_col+self.patch_scale[1]] \
            = patch_mask[tf_row:tf_row+self.patch_scale[0], tf_col:tf_col+self.patch_scale[1]]
        
        return img, mask
   
    
    def __call__(self, results):
        if 'cutmix' not in results:
            cutmix = True if np.random.rand() < self.prob else False
            results['cutmix'] = cutmix
        if 'cutmix_scale' not in results:
            results['cutmix_scale'] = self.patch_scale
        if results['cutmix']:
            img_name_list = glob.glob(results['img_prefix']+'/*')
            patch_image_path = random.choice(img_name_list)
            patch_mask_path = patch_image_path.replace('images', 'annotations').replace('jpg', 'png')
            patch_image = np.array(Image.open(patch_image_path))[:, :, ::-1]
            patch_mask = np.array(Image.open(patch_mask_path)) 
            
            tf_row_maxid = results['img_shape'][0] - self.patch_scale[0]
            tf_col_maxid = results['img_shape'][1] - self.patch_scale[1]
            
            img, mask = self.make_cutmix(results['img'], patch_image, results['gt_semantic_seg'], patch_mask, tf_row_maxid, tf_col_maxid)
            results['img'] = img
            for key in results.get('seg_fields', []):
                results[key] = mask
            
        return results

    
    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob}, patch_scale={self.patch_scale})'