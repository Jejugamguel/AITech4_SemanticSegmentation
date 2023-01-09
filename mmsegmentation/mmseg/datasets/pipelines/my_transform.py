import glob
import numpy as np
from numpy import random
from PIL import Image
from ..builder import PIPELINES

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
            patch_mask_path = patch_image_path.replace('train', 'train_mask').replace('jpg', 'png')
            # patch_mask_path = patch_image_path.replace('images', 'mask').replace('jpg', 'png')
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