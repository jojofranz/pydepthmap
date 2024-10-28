import sys
import os

import numpy as np

import scipy.ndimage
import scipy.stats
from scipy.ndimage import distance_transform_edt
import tifffile
import skimage
from skimage.io import imread
from skimage.util import view_as_windows
from skimage.filters import sato, median, gaussian
from skimage.filters.rank import percentile, maximum
from skimage.restoration import rolling_ball
from skimage.morphology import disk, ball, opening, area_opening, closing, binary_erosion, binary_dilation
from skimage.morphology import skeletonize
from scipy.ndimage import zoom
import cv2
import porespy as ps

import nibabel as nb

import dask
from dask import delayed
import dask.array as da
from dask.array import map_overlap
from dask_image import ndinterp
import dask.dataframe as dd

import zarr
from tqdm import tqdm
import pandas as pd
import copy

import cupy
from cupyx.scipy.ndimage import percentile_filter, grey_opening
import cupyx.scipy.ndimage as cpndi



def custom_se(radius, mode, spacing=10):
    """Create a sparse, custom structuring element for morphological operations."""
    from skimage.morphology import binary_erosion

    if mode == 'full':
        se = disk(radius)

    # successive circles
    if mode == 'sc':
        rings = np.arange(radius, 0, -spacing)
        se = disk(rings[0])
        se = se - binary_erosion(se)
        mid = (se.shape[0] // 2)

        for r in rings[1:]:
            fill = np.zeros_like(se)
            d_i = disk(r)
            d_i = d_i - binary_erosion(d_i)
            start = mid - (d_i.shape[0] // 2)
            stop = mid + (d_i.shape[0] // 2)
            fill[start:stop +1, start:stop +1] = d_i
            fill = fill ^ binary_erosion(fill)
            se = np.logical_or(se, fill)

    # regular pattern
    if mode == 'regp':
        se = disk(radius)
        pattern = np.zeros_like(se)
        for x in range(radius *2):
            for y in range(radius *2):
                if x% 2 == 0:
                    if y % 2 == 0:
                        pattern[x, y] = 1
                else:
                    if y % 2 == 1:
                        pattern[x, y] = 1
        se = np.logical_and(se, pattern)

    # random pattern
    if mode == 'randp':
        se = disk(radius)
        p = .2
        s = int(np.sum(se))
        n = int(np.round(s * p))
        pattern = np.random.permutation(np.hstack([np.ones(n), np.zeros(s - n)]))
        se[se == 1] = pattern

    se = se.astype('uint8')
    se = se > 0

    return se


def cv2_rm_background(im, se=disk(50), return_background=False, run_int8=False):
    # cv2 seems to interpret np.dtype('>u2') as uint8
    # safeguard against that
    if im.dtype == np.dtype('>u2'):
        im = im.astype(np.uint16)
        print('casting from >u2 to uint16')

    se = se.astype(np.uint8)
    import cv2
    if run_int8:
        dtype = im.dtype
        opening = im / (np.iinfo(dtype).max / 255)
        opening8 = opening.astype(np.uint8)
        opening8 = cv2.morphologyEx(opening8, cv2.MORPH_OPEN, se)
        opening = opening8 * (np.iinfo(dtype).max / 255)
        opening = opening.astype(dtype)
    else:
        opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, se)

    # from skimage.morphology import opening as grey_opening
    # opening = np.copy(im)
    # for slc in range(opening.shape[0]):
    #    opening[slc, ...] = grey_opening(im[slc, ...], se)

    # import dask
    # from dask_image import ndmorph
    # opening = np.copy(im)
    # for slc in range(opening.shape[0]):
    #    opening[slc, ...] = ndmorph.scipy.ndimage.grey_opening(im[slc, ...], footprint=se)

    background = np.minimum(im, opening)

    im = im - background
    if return_background:
        return im, background
    else:
        return im


def sparse_eq(im,
              stride,
              se=disk(40),
              percentile=90,
              return_p=False,
              pad=True,
              mode='cupy'):
    '''
    Equalization by division with slow, undesired intensity bias.
    These are estimated by the (sparse,) local percentile.

    Inspired by ClearMap.

    TODO: Implement striding in z,for 3D volumes.
          For now, the sparse grid is applied on each 2D image in a volume.

    im : 2D or 3D np.array
        The input image/volume
    stride : int
        The local percentile is calculated sparsely for each point on a grid
        where stride gives the distance in y and x between points.
    se : 2D array
        Structuring element, i.e. the mask applied at each point of the grid before
        calculating the local percentile for that point.
    percentile : scalar
        Percentile used to calculate the intensity bias.
    return_p : bool
        Whether tu return the estimate of the intensity bias.
    pad : bool
        Whether to pad the image (such that it stays the same size).
        Padding is done using the 'constant' mode and over a width that
        is half the shape of the structuring element.
        If False, make sure to pass an already padded image.
    mode : string
        'cupy' to use cupy/GPU processing
        'numpy' to use CPU processing

    Note: Regardless of the value of 'pad', padding is always
          removed from the output image. This reflects the assumption that if pad==False,
          the input image was padded beforehand. (This is more convenient when working with dask.)

    '''
    overlap_pad = np.array(se.shape) // 2
    if pad:
        im = np.pad(im, overlap_pad, mode='edge')

    ndim = im.ndim

    # 3D volumes that are really a 2D image
    # useful because dask often passes images from 3D volumes as 2D volumes of shape (1, y, x)
    treat_as_2d = False
    if ndim == 3 and im.shape[0] == 1:
        im = im[0, ...]
        treat_as_2d = True

    window_shape = se.shape
    B = view_as_windows(im, window_shape, stride)
    se_n = np.sum(se)

    grid_shape = B.shape
    data = []
    if ndim == 3:
        for z in range(grid_shape[0]):
            for y in range(grid_shape[1]):
                for x in range(grid_shape[2]):
                    data.append(B[z, y, x][se])
        data = np.stack(data)
        data = data.reshape((grid_shape[0], grid_shape[1], grid_shape[2], se_n))
    else:
        for y in range(grid_shape[0]):
            for x in range(grid_shape[1]):
                data.append(B[y, x][se])
        data = np.stack(data)
        data = data.reshape((grid_shape[0], grid_shape[1], se_n))

    if mode == 'cupy':
        B = cupy.asarray(data)
        p = cupy.percentile(B, percentile, axis=-1)
        p = cpndi.zoom(p, stride, order=1, mode='nearest')
        p = cupy.asnumpy(p)
    elif mode == 'numpy':
        p = np.percentile(data, percentile, axis=-1)
        p = zoom(p, stride)

    if ndim == 3:
        im = im[overlap_pad[0]:-overlap_pad[0],
             overlap_pad[1]:-overlap_pad[1],
             overlap_pad[2]:-overlap_pad[2]]
    else:
        im = im[overlap_pad[0]:-overlap_pad[0],
             overlap_pad[1]:-overlap_pad[1]]
    im = im / p
    # TODO: change this - it assumes that after division, np.max(im) <= 10
    # and will result in overflow otherwise
    im = (im * (np.iinfo(np.uint16).max / 10)).astype(np.uint16)

    if treat_as_2d:
        im = im[None, ...]
        p = p[None, ...]

    if not return_p:
        return im
    else:
        return im, p


def preproc_angio_for_vis(im):
    '''
    preprocess a 2D angio image for visualization by
    1) equalization with
            - sparse structuring element (successive circles) with
              radius = 40
            - stride = 20
            - percentile = 1
    '''
    radius = 40
    window_shape = (radius * 2 + 1,
                    radius * 2 + 1)
    overlap_pad = window_shape[0] // 2

    se = custom_se(radius, 'sc')
    stride = 20
    eq, p = sparse_eq(im,
                      stride,
                      se=se,
                      percentile=1,
                      return_p=True,
                      mode='cupy')
    return eq


def preproc_cyto_for_vis(im):
    '''
    preprocess a 2D cyto image for visualization by
    1) background removal with
            - full structuring element with
              radius = 40
    2) equalization
            - sparse structuring element (successive circles) with
              radius = 40
            - stride = 20
            - percentile = 90
    '''
    radius = 40
    window_shape = (radius * 2 + 1,
                    radius * 2 + 1)
    overlap_pad = window_shape[0] // 2
    se = custom_se(radius, 'full')
    stride = 20

    brm, b = cv2_rm_background(im,
                               se=se,
                               return_background=True,
                               run_int8=False)

    se = custom_se(radius, 'sc')

    eq, p = sparse_eq(brm,
                      stride,
                      se=se,
                      percentile=90,
                      return_p=True,
                      mode='cupy')

    return eq