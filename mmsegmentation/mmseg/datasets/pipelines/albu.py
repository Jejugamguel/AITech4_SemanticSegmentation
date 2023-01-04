import numpy as np
import cv2
import albumentations as A
from numpy import random
from ..builder import PIPELINES



@PIPELINES.register_module()
class A_CLAHE(object):
    """
    Apply Contrast Limited Adaptive Histogram Equalization to the input image.
    
    input : dict
    output : dict
    
    Args:
        prob (float, optional) : probability of applying ChannelShuffle
        clip_limit (float or [float, float]) :upper threshold value for contrast limiting. If clip_limit is a single float value, the range will be (1, clip_limit). Default: (1, 4).
        tile_grid_size ([int, int]) : size of grid for histogram equalization. Default: (8, 8).
    """
    def __init__(self,prob,clip_limit=4.0, tile_grid_size=(32, 32)):
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


# @PIPELINES.register_module()
# class CLAHE(object):
#     """
#     Apply Contrast Limited Adaptive Histogram Equalization to the input image.
    
#     input : dict
#     output : dict
    
#     Args:
#         prob (float, optional) : probability of applying ChannelShuffle
#         clip_limit (float or [float, float]) :upper threshold value for contrast limiting. If clip_limit is a single float value, the range will be (1, clip_limit). Default: (1, 4).
#         tile_grid_size ([int, int]) : size of grid for histogram equalization. Default: (8, 8).
#     """
#     def __init__(self,prob,clip_limit=4.0, tile_grid_size=(32, 32)):
#         self.prob = prob
#         self.clip_limit= clip_limit
#         self.tile_grid_size = tile_grid_size
#         if prob is not None:
#             assert prob >= 0 and prob <= 1



#     def __call__(self, results):
#         if 'clahe' not in results:
#             clahe = True if np.random.rand() < self.prob else False
#             results['clahe'] = clahe

#         if results['clahe']:
#             image = results['img']
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             mask = results['gt_semantic_seg']

#             transform = A.Compose([
#                         A.CLAHE(clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size,p=self.prob),
#                         ],
#                         p=1)
#             transformed = transform(image=image, mask=mask)
#             results['img'] = transformed['image']
#             results['gt_semantic_seg'] = transformed['mask']
#         return results

    
#     def __repr__(self):
#         return self.__class__.__name__ + f'(prob={self.prob})'




# @PIPELINES.register_module()
# class Transpose2(object):
#     """
#     Transpose the input by swapping rows and columns.
    
#     input : dict
#     output : dict
    
#     Args:
#         prob (float, optional) : probability of applying ChannelShuffle
        
#     """
#     def __init__(self,prob):
#         self.prob = prob

#         if prob is not None:
#             assert prob >= 0 and prob <= 1




#     def __call__(self, results):
#         if 'transpose' not in results:
#             transpose = True if np.random.rand() < self.prob else False
#             results['transpose'] = transpose

#         if results['transpose']:
#             image = results['img']
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             mask = results['gt_semantic_seg']

#             transform = A.Compose([
#                         A.Transpose(p=self.prob),
#                         ],
#                         p=1)
#             transformed = transform(image=image, mask=mask )
#             results['img'] = transformed['image']
#             results['gt_semantic_seg'] = transformed['mask']
#         return results

    
#     def __repr__(self):
#         return self.__class__.__name__ + f'(prob={self.prob})'



@PIPELINES.register_module()
class ElasticTransform(object):
    """
    Blur the input image using a Gaussian filter with a random kernel size.
    
    input : dict
    output : dict
    
    Args:
        prob (float, optional) : probability of applying ChannelShuffle
        
    """
    def __init__(self,prob,alpha=1, sigma=50,alpha_affine=50, interpolation=1,\
        border_mode=4, value=None, mask_value=None, \
        approximate=False, same_dxdy=False):
        self.prob = prob
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.approximate = approximate
        self.same_dxdy = same_dxdy

        if prob is not None:
            assert prob >= 0 and prob <= 1




    def __call__(self, results):
        if 'ElasticTransform' not in results:
            transpose = True if np.random.rand() < self.prob else False
            results['ElasticTransform'] = transpose

        if results['ElasticTransform']:
            image = results['img']
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = results['gt_semantic_seg']

            transform = A.Compose([
                        A.ElasticTransform(p=self.prob),
                        ],
                        p=1)
            transformed = transform(image=image, mask=mask )
            results['img'] = transformed['image']
            results['gt_semantic_seg'] = transformed['mask']
        return results

    
    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class SafeRotate(object):
    """
    Blur the input image using a Gaussian filter with a random kernel size.
    
    input : dict
    output : dict
    
    Args:
        prob (float, optional) : probability of applying ChannelShuffle
        
    """
    def __init__(self,prob,limit=90, interpolation=1,border_mode=4, value=None, mask_value=None):
        self.prob = prob
        self.limit = limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

        if prob is not None:
            assert prob >= 0 and prob <= 1




    def __call__(self, results):
        if 'saferotate' not in results:
            transpose = True if np.random.rand() < self.prob else False
            results['saferotate'] = transpose

        if results['saferotate']:
            image = results['img']
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = results['gt_semantic_seg']

            transform = A.Compose([
                        A.SafeRotate(p=self.prob,limit=self.limit,interpolation=self.interpolation,border_mode=self.border_mode,
                        value=self.value,mask_value=self.mask_value),
                        ],
                        p=1)
            transformed = transform(image=image, mask=mask )
            results['img'] = transformed['image']
            results['gt_semantic_seg'] = transformed['mask']
        return results

    
    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'



@PIPELINES.register_module()
class FancyPCA(object):
    """
    Augment RGB image using FancyPCA from Krizhevsky's paper "ImageNet Classification with Deep Convolutional Neural Networks"
    
    input : dict
    output : dict
    
    Args:
        alpha	 (float) : 	how much to perturb/scale the eigen vecs and vals. scale is samples from gaussian distribution (mu=0, sigma=alpha)
        
    """
    def __init__(self,prob,alpha=0.1):
        self.prob = prob
        self.alpha = alpha
        if prob is not None:
            assert prob >= 0 and prob <= 1




    def __call__(self, results):
        if 'fancyPCA' not in results:
            transpose = True if np.random.rand() < self.prob else False
            results['fancyPCA'] = transpose

        if results['fancyPCA']:
            image = results['img']
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = results['gt_semantic_seg']

            transform = A.Compose([
                        A.FancyPCA(p=self.prob,alpha=self.alpha),
                        ],
                        p=1)
            transformed = transform(image=image, mask=mask )
            results['img'] = transformed['image']
            results['gt_semantic_seg'] = transformed['mask']
        return results

    
    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


