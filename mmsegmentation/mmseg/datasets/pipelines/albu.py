import numpy as np
import cv2
import albumentations as A
from numpy import random
from ..builder import PIPELINES

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