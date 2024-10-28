import sys
import os

import numpy as np

import hist

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

import subprocess

import math

import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

import edt

import cupy
from cupyx.scipy.ndimage import percentile_filter, grey_opening
import cupyx.scipy.ndimage as cpndi

from .utils import get_LayNii
#from .visualization import plot_layers


def get_manual_layers(index, image, mip_roi, 
                      lr_index_data, rim_data, 
                      lr_physical_voxel_size_um, 
                      manual_layer_im_fn, 
                      manual_layer_boundaries_fn, 
                      manual_layer_names, 
                      pad=(0, 0, 0, 0), 
                      plot=False):
    """Get the manual layer definitions (with bells and whistles).
    Manual layer definition was done on a screenshot of a z-MIP of parts of the data.
    As an added complication, the screenshot might have been a cropped + zoomed-in/out version of the MIP.
    Here, we want to extract the relative depth [0, 1], as well as physical depth [mm] for 
    the layer boundaries, as well as their average position.

    The approach here was to extract a point per layer boundary from the screenshot.
    This was done using LabKit as a point-annotation tool.
    Specifically, one point (single foreground label-pixel) was placed for each boundary except for the pial surface.
    The points were placed in order from superficial to deep.

    Then the points are transferred from the reference space of the screenshot to the reference space of the data.
    Then they can be used to extract the relative and physical depth of the layer boundaries.

    On a side-note, this also extracts the physical depth at 20, 40, 60, 80% of the cortical depth.

    Parameters
    ----------
    index : array
        The index array (full resolution, after potential cropping)
    image : array
        The image array (full resolution, after potential cropping)
    mip_roi : 3-tuple of slice
        The ROI for the MIP
        NOTE: currently, only z-slicing is implemented (no stepping)
    lr_index_data : array
        The low-resolution index data (metric obtained with Laynii)
    rim_data : array
        The rim data (rim input to Laynii)
    lr_physical_voxel_size_um : array
        The physical voxel size of the low-resolution data
    manual_layer_im_fn : str
        The filename of the screenshot of the MIP
    manual_layer_boundaries_fn : str
        The filename of the manual layer boundaries (the Labkit labeling file)
    manual_layer_names : list
        The names of the manual layers
    pad : 4-tuple of int
        The padding applied (to index,image,lr_index_data,rim_data) since taking the MIP
    plot : bool
        Whether to plot the results

    Returns
    -------
    manual_layers : dict
        A dictionary with the manual layer definitions
        - manual_layer_boundaries_Rdepth : list
            The relative depth of the layer boundaries
        - manual_layer_boundary_Pdepths : list
            The physical depth of the layer boundaries
        - manual_layer_names : list
            The names of the manual layers
        - manual_layer_Rdepth : list
            The average relative depth of the layers
        - manual_layer_Pdepths : list
            The average physical depth of the layers
        - manual_layer_width_Rdepth : list
            The width of the layers in relative depth
        - manual_layer_width_Pdepths : list
            The width of the layers in physical depth
        - depth_indicator_Rdepths : list
            The relative depth levels
        - depth_indicator_Pdepths : list
            The physical depth levels
    
    """
    
    assert np.all([slc.step is None for slc in mip_roi]), 'Non-zero steps not implemented'
    assert np.all([slc.start is None for slc in mip_roi[1:]]), 'Slicing only implemented along z'
    assert np.all([slc.stop is None for slc in mip_roi[1:]]), 'Slicing only implemented along z'

    # layers were manually defined on a MIP of parts of the data
    im = np.max(image[mip_roi], axis=0).compute().astype(float)

    # also, some cropping was done after the manual definition
    # add back the padding
    im = np.pad(im, ((pad[0], pad[1]), (pad[2], pad[3])), 
                mode='constant', constant_values=np.nan)

    # actually, they were defined on a screenshot of that MIP
    # (i.e. on a scaled/zoomed version)
    manual_layer_im = skimage.io.imread(manual_layer_im_fn)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        #img0 = ax.imshow(im, vmin = 2000, vmax = 13000, cmap='Reds')
        #ax.imshow(manual_layer_im, alpha=.5, extent=img0.get_extent())

    # find back the zoom-factor
    scale = im.shape[0] / manual_layer_im.shape[0]

    #manual_layer_boundaries_fn = 'manual_layer_def/new/v1_layers2.0.PNG.labeling'
    #manual_layer_names = ['I', 'II', 'III', 'IVa', 'IVb', 'IVc', 'V', 'VIa', 'VIb']

    import json
    with open(manual_layer_boundaries_fn) as f:
        manual_layer_boundaries = f.readlines()
    manual_layer_boundaries = json.loads(manual_layer_boundaries[0])

    manual_layer_boundaries = manual_layer_boundaries['labels']['foreground']
    manual_layer_boundaries = np.array(manual_layer_boundaries)
    manual_layer_boundaries = (manual_layer_boundaries * scale).astype(np.uint16)
    # remove the point at wm, replace by 0 below
    manual_layer_boundaries = manual_layer_boundaries[:len(manual_layer_names)-1]

    metric = index
    metric = metric[mip_roi[0].start + 
                    int((mip_roi[0].stop - mip_roi[0].start)/2), ...].compute().astype(float)
    # account for the cropping
    metric = np.pad(metric, ((pad[0], pad[1]), (pad[2], pad[3])), mode='constant', constant_values=np.nan)
    manual_layer_boundaries_Rdepths = [metric[x, y] for (y, x) in manual_layer_boundaries]
    manual_layer_boundaries_Rdepths.insert(0, 1)
    manual_layer_boundaries_Rdepths.insert(len(manual_layer_boundaries_Rdepths), 0)

    if plot:
        # ugly, prevent circular import error
        from .visualization import plot_layers
        plot_layers(metric, manual_layer_boundaries_Rdepths, ax=ax, alpha=.2, cmap='binary')

    # get physical depth for each value in manual_layer_boundaries_Rdepth
    manual_layer_boundary_Pdepths = get_average_depths(rim_data, lr_index_data, manual_layer_boundaries_Rdepths, physical_voxel_size_um=lr_physical_voxel_size_um)
    # or, get physical depth for each bin in manual_layer_boundaries_Rdepth
    manual_layer_Pdepths = get_average_layer_depths(rim_data, lr_index_data, manual_layer_boundaries_Rdepths, physical_voxel_size_um=lr_physical_voxel_size_um)
    manual_layer_Pdepths = np.round(manual_layer_Pdepths, 2)

    manual_layer_Rdepths = get_avg_bin_loc(manual_layer_boundaries_Rdepths)

    # get physical depth at 20,40,60,80%
    depth_indicator_Rdepths = [.2, .4, .6, .8]
    depth_indicator_Pdepths = get_average_depths(rim_data, lr_index_data, depth_indicator_Rdepths, physical_voxel_size_um=lr_physical_voxel_size_um)
    depth_indicator_Pdepths = np.round(depth_indicator_Pdepths, 1)


    manual_layer_boundaries_Rdepths = np.array(manual_layer_boundaries_Rdepths)
    manual_layer_boundary_Pdepths = np.array(manual_layer_boundary_Pdepths)
    manual_layer_names = np.array(manual_layer_names)
    manual_layer_Rdepths = np.array(manual_layer_Rdepths)
    manual_layer_Pdepths = np.array(manual_layer_Pdepths)
    depth_indicator_Rdepths = np.array(depth_indicator_Rdepths)
    depth_indicator_Pdepths = np.array(depth_indicator_Pdepths)

    manual_layers = {'manual_layer_boundary_Rdepths': manual_layer_boundaries_Rdepths, 
                     'manual_layer_boundary_Pdepths': manual_layer_boundary_Pdepths, 
                     'manual_layer_names': manual_layer_names,
                     'manual_layer_Rdepths': manual_layer_Rdepths, 
                     'manual_layer_Pdepths': manual_layer_Pdepths, 
                     'manual_layer_width_Rdepths': np.array(manual_layer_boundaries_Rdepths[:-1]) - np.array(manual_layer_boundaries_Rdepths[1:]),
                     'manual_layer_width_Pdepths': np.array(manual_layer_boundary_Pdepths[1:]) - np.array(manual_layer_boundary_Pdepths[:-1]),
                     'depth_indicator_Rdepths': depth_indicator_Rdepths, 
                     'depth_indicator_Pdepths': depth_indicator_Pdepths}
        
    return manual_layers



def find_layer_peaks(angio_profile, cyto_profile, angio_color, cyto_color, average_layer_depths, 
                     angio_min_peak_height=10, cyto_min_peak_height=50000, 
                     angio_min_peak_dist=20, cyto_min_peak_dist=5, 
                     sigma=2, show=True):
    """Find peaks in angio- and cyto-profiles and extract their depths (in mm).
    The profiles are smoothed with a Gaussian filter before peak detection.

    Parameters
    ----------
    angio_profile : array
        The angio profile
    cyto_profile : array
        The cyto profile
    angio_color : tuple
        The color of the angio profile
    cyto_color : tuple
        The color of the cyto profile
    average_layer_depths : array
        The average layer depths
        must be same length as angio_profile and cyto_profile
    angio_min_peak_height : int
        The minimum peak height in the angio profile
    cyto_min_peak_height : int
        The minimum peak height in the cyto profile
    angio_min_peak_dist : int
        The minimum peak distance in the angio profile
    cyto_min_peak_dist : int
        The minimum peak distance in the cyto profile
    sigma : int
        The sigma for the gaussian filter
    show : bool
        Whether to show the profiles

    Returns
    -------
    peaks : DataFrame
        The peaks and their depths and colors
        
    """
    
    n_depthbins = len(angio_profile)
    
    s_v = gaussian_filter1d(angio_profile, sigma)
    p_v = find_peaks(s_v, angio_min_peak_height, distance=angio_min_peak_dist)

    s_c = gaussian_filter1d(cyto_profile, sigma)
    p_c = find_peaks(s_c, cyto_min_peak_height, distance=cyto_min_peak_dist)

    p = np.sort(np.hstack([p_v[0], p_c[0]]))
    lbl = ['cyto' if i in p_c[0] else 'angio' for i in p]
    c = [cyto_color if i in p_c[0] else angio_color for i in p]

    labels = [average_layer_depths[::-1][i] for i in p]
    labels = list(np.round(np.array(labels), 2))

    # black for 'WM' and 'GM' label
    c.insert(0, (0, 0, 0, 1))
    c.insert(len(c), (0, 0, 0, 1))
    p = np.hstack([0, p, n_depthbins])
    labels = np.hstack(['WM', labels, 'CSF'])

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(7, 7))
        ax[0].plot(angio_profile, 
                np.arange(n_depthbins), 
                c=angio_color)
        ax[0].plot(s_v, 
                np.arange(n_depthbins), 
                c='k')


        ax[1].plot(cyto_profile, 
                np.arange(n_depthbins), 
                c=cyto_color)
        ax[1].plot(s_c, 
                np.arange(n_depthbins), 
                c='k')

        ax[0].set_yticks(p)
        ax[0].set_yticklabels(labels)#, fontdict=[{'color': purple(100)}, {'color': green(100)}])
        for ytick, color in zip(ax[0].get_yticklabels(), c):
            ytick.set_color(color)

        ax[1].set_yticks(p)
        ax[1].set_yticklabels(labels)#, fontdict=[{'color': purple(100)}, {'color': green(100)}])
        for ytick, color in zip(ax[1].get_yticklabels(), c):
            ytick.set_color(color)
        
    return pd.DataFrame({'peak': p[1:-1], 'label': lbl, 'color': c[1:-1]})
    


def volume_from_cellcenters(cellcenters, data, fn, mask=None):
    """Generate a binary volume from cell centers.
    Generate a volume with the same shape as data, where the cell centers are set to True.
    If a mask is provided, only the cell centers OUTSIDE the mask are set to True.

    Parameters
    ----------
    cellcenters : array
        The cell centers, shape (n_cells, 3)
    data : array
        The data array
    fn : str
        The filename to save the volume to
    mask : array
        The mask array

    Returns
    -------
    cellcenters : array
        The cell centers (outside the mask if a mask is provided)
    """
    shape = data.shape
    dtype = data.dtype
    chunksize = data.chunksize
    # TODO: this creates a file even if it fails later
    store = zarr.DirectoryStore(fn)
    volume = zarr.create(shape, 
                         chunks=chunksize, 
                         dtype=bool, 
                         store=store, 
                         fill_value=False)

    # looks like there are very few repeated cell locations in the stardist output
    # avoid indexing errors 
    cellcenters = np.unique(cellcenters, axis=0)
    cellcenters = cellcenters.astype(int)
    
    # only take cellcenters that are inside the volume
    inside = np.all(np.stack([cc < s for cc, s in zip(cellcenters.T, shape)]).T, axis=1)
    cellcenters = cellcenters[inside]

    # at this point, cellcenters might not be in the same order as the volume
    # i.e. np.where(A > 0) normally corresponds to A[A > 0], but here np.where(A > 0) (the cellcenters) might be permuted
    # re-arrange cellcenters to match the volume (to gain correspondence with cellcenters_inside_mask)
    # (this correspondence is important downstream when assigning cellcenters_outside_mask but nice to have in general)
    cellcenters = tuple([dim for dim in cellcenters.T])
    linear_indices = np.ravel_multi_index(cellcenters, shape)
    argsort = np.argsort(linear_indices)
    cellcenters = np.stack([index[argsort] for index in cellcenters]).T
    # TODO could make this more scalable
    volume[cellcenters[:, 0], cellcenters[:, 1], cellcenters[:, 2]] = True

    ## DEMO: re-order a permuted version of np.where(A > 0) to correspond to A[A > 0]
    ## Example array
    #A = np.random.choice([0, 0, 1, 4], (4, 4, 4))
    #print("A:", A)
    ## Original indices from np.where(A > 0)
    #original_indices = np.where(A > 0)
    ## Simulate permuted indices (for example)
    #permute = np.random.permutation(np.arange(len(original_indices[0])))
    #permuted_indices = tuple([dim[permute] for dim in original_indices])
    ## Convert the original and permuted indices to linear indices
    #original_linear_indices = np.ravel_multi_index(original_indices, A.shape)
    #permuted_linear_indices = np.ravel_multi_index(permuted_indices, A.shape)
    ## Sort the permuted indices based on the original order
    #sorted_index_order = np.argsort(permuted_linear_indices)
    #sorted_permuted_indices = tuple(index[sorted_index_order] for index in permuted_indices)
    ## Verify correspondence
    #original_values = A[original_indices]  # These are the values in the original order
    #restored_values = A[sorted_permuted_indices]  # These should match the original values
    #print("Original values:", original_values)
    #print("Restored values:", restored_values)

    if mask is not None:
        # TODO make this more scalable
        # 1D index of which cellcenters are inside the mask
        cellcenters_inside_mask = mask[da.from_zarr(volume)].compute()        
        cellcenters_outside_mask = np.logical_not(cellcenters_inside_mask)
        volume[cellcenters[:, 0], cellcenters[:, 1], cellcenters[:, 2]] = cellcenters_outside_mask
        return cellcenters[cellcenters_outside_mask, :]
    else:
        return cellcenters


def postprocess_density(density, depthbins, binary_data=True, databins=None):
    """Extract summary statistics from a density DataFrame.

    Parameters
    ----------
    density : DataFrame
        The density DataFrame with columns:
        - index : int
            index through the indexbins
        - indexbin_volume : int
            volume of the indexbin
            NOTE: this is repeated for all rows of a 'index'x'chunk_id' combination
            that is for the different 'value' rows of each 'index'x'chunk_id' combination
        - chunk_id : int
            index through the chunks of the input dask-arrays
        - value : int
            index throught the databins (e.g. layers)
        - count : int
            number of voxels in the gived indexbin-by-chunk_id-by-value combination

    Returns
    -------
    density_per_indexbin : DataFrame
        The density per indexbin
    density_per_indexbin_per_chunk : DataFrame
        The density per indexbin per chunk

    Notes
    -----
    SEMs in the output DataFrames are computed over chunk_ids
    """
    if not np.all(np.diff(depthbins) > 0):
        raise Exception('depthbins need to be strictly increasing')
    
    density['index_bin_start'] = density['index'].apply(lambda x: depthbins[x])
    density['index_bin_end'] = density['index'].apply(lambda x: depthbins[x+1])

    if not binary_data:
        if databins is None:
            raise ValueError('databins should be provided if binary_data is False')
        density['databin_start'] = density['value'].apply(lambda x: databins[x])
        density['databin_end'] = density['value'].apply(lambda x: databins[x+1])

    # sum over all chunks
    # NOTE: 'indexbin_volume' is repeated for all rows of a 'index'x'chunk_id' combination
    # that is for the different 'value' rows of each 'index'x'chunk_id' combination
    # ignore that by using max
    indexbin_volume = density.groupby(['index', 'chunk_id'])['indexbin_volume'].max().groupby(['index']).sum()
    percent_indexbin_volume = indexbin_volume / np.sum(indexbin_volume) * 100

    if binary_data:
        print('binary data')
        density = density.loc[density['value'] == 1]

    foreground_volume_per_indexbin = density.groupby('index').apply(lambda x: np.sum(x['count']))
    density_per_indexbin = foreground_volume_per_indexbin / indexbin_volume
    percent_density_per_indexbin = density_per_indexbin * 100

    count_per_indexbin_per_chunk = density.groupby(['index', 'chunk_id'])['count'].sum()
    count_per_indexbin_per_chunk = count_per_indexbin_per_chunk.to_frame().reset_index()
    indexbin_volume_per_chunk = density.groupby(['index', 'chunk_id'])['indexbin_volume'].max()
    indexbin_volume_per_chunk = indexbin_volume_per_chunk.to_frame().reset_index()
    for_density_per_indexbin_sem = pd.merge(count_per_indexbin_per_chunk, indexbin_volume_per_chunk, on=['index', 'chunk_id'])
    density_per_indexbin_sem = for_density_per_indexbin_sem.groupby(['index']).apply(lambda x: scipy.stats.sem(np.divide(x['count'], 
                                                                                                                        x['indexbin_volume'])))
    percent_density_per_indexbin_sem = for_density_per_indexbin_sem.groupby(['index']).apply(lambda x: scipy.stats.sem(np.divide(x['count'], 
                                                                                                                                 x['indexbin_volume'])*100))
    density_per_indexbin = pd.DataFrame({'index_bin': np.arange(len(depthbins) - 1), 
                                         'index_bin_start': depthbins[:-1], 
                                         'index_bin_end': depthbins[1:], 
                                         'indexbin_volume': indexbin_volume, 
                                         'data_volume': foreground_volume_per_indexbin, 
                                         'density': density_per_indexbin, 
                                         'density_sem': density_per_indexbin_sem, 
                                         'percent_density': percent_density_per_indexbin, 
                                         'percent_indexbin_volume': percent_indexbin_volume, 
                                         'percent_density_sem': percent_density_per_indexbin_sem})
    
    count_per_indexbin_per_value = density.groupby(['index', 'value'], 
                                                as_index=False).apply(lambda x: np.sum(x['count']))
    count_per_indexbin_per_value = count_per_indexbin_per_value.rename(columns={None: 'count'})

    indexbin_volume = density.groupby(['index', 'chunk_id'])['indexbin_volume'].max().groupby(['index']).sum()
    indexbin_volume = indexbin_volume.to_frame().reset_index()

    density_per_indexbin_per_value = pd.merge(count_per_indexbin_per_value, indexbin_volume, on=['index'])
    percent_density_per_indexbin_per_value = pd.DataFrame({'index': density_per_indexbin_per_value['index'], 
                                                           'value': density_per_indexbin_per_value['value'], 
                                                           'density': density_per_indexbin_per_value['count'] / 
                                                                      density_per_indexbin_per_value['indexbin_volume'] * 100})
    percent_density_per_indexbin_per_value = percent_density_per_indexbin_per_value.pivot(index='index', 
                                                                                          columns='value', 
                                                                                          values='density')
    
    # also create percent_density_per_indexbin_per_chunk
    count_per_indexbin_per_chunk = density.groupby(['index', 'chunk_id'], 
                                                  as_index=False).apply(lambda x: np.sum(x['count']))
    count_per_indexbin_per_chunk = count_per_indexbin_per_chunk.rename(columns={None: 'count'})

    indexbin_volume_per_chunk = density.groupby(['index', 'chunk_id'])['indexbin_volume'].max()
    indexbin_volume_per_chunk = indexbin_volume_per_chunk.to_frame().reset_index()

    density_per_indexbin_per_chunk = pd.merge(count_per_indexbin_per_chunk, indexbin_volume_per_chunk, on=['index', 'chunk_id'])
    percent_density_per_indexbin_per_chunk = pd.DataFrame({'index': density_per_indexbin_per_chunk['index'], 
                                                           'chunk_id': density_per_indexbin_per_chunk['chunk_id'], 
                                                           'density': density_per_indexbin_per_chunk['count'] / 
                                                                      density_per_indexbin_per_chunk['indexbin_volume'] * 100})
    percent_density_per_indexbin_per_chunk = percent_density_per_indexbin_per_chunk.pivot(index='index', 
                                                                                          columns='chunk_id', 
                                                                                          values='density')
    
    return [density_per_indexbin, 
            percent_density_per_indexbin_per_value, 
            percent_density_per_indexbin_per_chunk]


def measure_hist(data, bins):
    if not np.all(np.diff(bins) > 0):
        raise Exception('bins need to be strictly increasing')
    data_blocks = data.to_delayed().ravel()
    hist = [dd.from_delayed(delayed_hist(i, data, bins), meta=delayed_hist_meta) 
            for i, data in enumerate(data_blocks)]
    hist_ddf = dd.concat(hist)
    hist_ddf = hist_ddf.compute()
    return hist_ddf

@dask.delayed
def delayed_hist(i, data, bins):
    hist, _ = np.histogram(data, bins=bins)
    return pd.DataFrame({'index': np.arange(len(bins)-1), 
                         'count': hist, 
                         'chunk_id': np.repeat(i, len(bins)-1)})

delayed_hist_meta = dd.utils.make_meta([('index', np.int64),
                                        ('count', np.int64),
                                        ('chunk_id', np.int64)])

def measure_density(data, index, indexbins, binary_data=True, 
                    databins=None, groupby_dim=None):
    """ A wrapper around delayed_indexed_hist to compute density per indexbin
    This wrapper takes care of making the right call for binary or non-binary data

    Parameters
    ----------
    data : dask.array
        The data array
        Can be a masked array (see indexed_hist)
    index : dask.array
        The index array
    indexbins : array
        The indexbins
    binary_data : bool
        Whether the data is binary
    databins : array, optional
        The databins
    groupby_dim : str, optional
        The dimension to groupby

    Returns
    -------
    density_ddf : DataFrame
        The density DataFrame
        
    """
    if not np.all(np.diff(indexbins) > 0):
        raise Exception('depthbins need to be strictly increasing')
    n_depthbins = len(indexbins) - 1
    data_blocks = data.to_delayed().ravel()
    index_blocks = index.to_delayed().ravel()
    if binary_data:
        if databins is not None:
            raise ValueError('databins should not be provided if binary_data is True')
        density = [dd.from_delayed(delayed_indexed_hist(i, data, index, indexbins, binary_data=True, background_val=.5, 
                                                        groupby_dim=groupby_dim), 
                                                        meta=[delayed_indexed_hist_gd_meta, 
                                                              delayed_indexed_hist_meta][groupby_dim is None]) 
                     for i, (index, data) 
                     in enumerate(zip(index_blocks, data_blocks))]
    else:
        density = [dd.from_delayed(delayed_indexed_hist(i, data, index, indexbins, binary_data=False, 
                                                        databins=databins, groupby_dim=groupby_dim), 
                                                        meta=[delayed_indexed_hist_gd_meta, 
                                                              delayed_indexed_hist_meta][groupby_dim is None]) 
                     for i, (index, data) 
                     in enumerate(zip(index_blocks, data_blocks))]
    density_ddf = dd.concat(density)
    density_ddf = density_ddf.compute()
#    density_ddf['layer'] = np.tile(np.arange(n_depthbins), 
#                                   len(np.unique(density_ddf['chunk_id'])))
    return density_ddf

def get_avg_bin_loc(arr):
    return [np.mean([arr[i], arr[i+1]]) for i in range(len(arr)-1)]

def get_average_depths(rim, metric_valid, depths, physical_voxel_size_um=[10, 10.3, 10.3]):
    """Get average cortical depth per depth
    compared to get_average_layer_depths, this function does not work with bins but with exact indices
    """
    physical_voxel_size_mm = np.array(physical_voxel_size_um) / 1000
    depth = edt.edt(rim != 1, anisotropy=physical_voxel_size_mm)  # distance from CSF
    cortex_valid = metric_valid != 0
    depths_depths = []
    for i in range(len(depths)):
        d = 0.001
        sufficient_values = False
        while not sufficient_values:
            histbin_mask = np.logical_and.reduce([cortex_valid, 
                                                metric_valid < depths[i] + d, 
                                                metric_valid >= depths[i] - d])
            if not np.any(histbin_mask):
                d *= 2
            else:
                sufficient_values = True
        depths_depths.append(np.mean(depth[histbin_mask]))
    return depths_depths

def get_average_layer_depths(rim, metric_valid, depthbins, physical_voxel_size_um=[10, 10.3, 10.3]):
    """Get average cortical depth per depthbin
    """
    physical_voxel_size_mm = np.array(physical_voxel_size_um) / 1000
    depth = edt.edt(rim != 1, anisotropy=physical_voxel_size_mm)  # distance from CSF
    cortex_valid = metric_valid != 0
    depthbins_depths = []
    for i in range(len(depthbins)-1):
        histbin_mask = np.logical_and.reduce([cortex_valid, 
                                              metric_valid < depthbins[i], 
                                              metric_valid >= depthbins[i+1]])
        depthbins_depths.append(np.mean(depth[histbin_mask]))
    return depthbins_depths

def dask_zoom(data, real_zoom, output_chunks, order=1):

    if isinstance(data, np.ndarray):
        data = da.from_array(data)

    if isinstance(output_chunks, int):
        output_chunks = (output_chunks, output_chunks, output_chunks)
    elif isinstance(output_chunks, list):
        if len(output_chunks) == 1:
            output_chunks = (output_chunks[0], output_chunks[0], output_chunks[0])
        elif len(output_chunks) == 3:
            output_chunks = tuple(output_chunks)
        else:
            raise Exception('output_chunks should be int or list of len 3')
    else:
        raise Exception('output_chunks should be int or list of len 3')
        
    # TODO: is float32 necessary and sufficient?
    data = data.astype(np.float32)
    data[data == 0] = np.nan
    if isinstance(real_zoom, int) or isinstance(real_zoom, float):
        real_zoom = [real_zoom, real_zoom, real_zoom]
    if len(real_zoom) == 1:
        real_zoom = [real_zoom[0], real_zoom[0], real_zoom[0]]
    if len(real_zoom) == 3:
        real_zoom = list(real_zoom)
    else:
        raise Exception('real_zoom should be int or list of len 3')
    real_zoom = np.array(real_zoom)
    # TODO: this is needed to avoid a 1-off error (last slice empty) - is there a better way?
    zoom_factor = (data.shape[0] * real_zoom) / (data.shape[0] - 1)
    zoom_factor = np.append(zoom_factor, 1)
    matrix = np.diagflat(1 / zoom_factor)
    output_shape = tuple(np.array(data.shape) * real_zoom)
    zoomed = ndinterp.affine_transform(data, 
                                       matrix=matrix, 
                                       output_shape=output_shape, 
                                       order=order, 
                                       output_chunks=output_chunks)

    return zoomed



def proc_metric_equidist(metric_equidist, proc, rim):
    """
    NOTE: This is very specific to the present dataset
    """
    print(f'Processing metric_equidist ... NOTE this is function is very specific to the present dataset - DO NOT USE IT ELSEWHERE!')
    # TODO: make this addition+removal of the border more explicit throughout
#    metric_equidist = metric_equidist[1:-1, 1:-1, 1:-1]
#    rim = rim[1:-1, 1:-1, 1:-1]

    # treat missing data
    missing = proc == 2
    # exclude positions that are close to missing WM and CSF references
    # WM
    WM = rim == 2
    # NOTE: this is very specific to the present dataset
    # Assuming that the bottom-most slice should be WM
    WM_missing = np.zeros_like(WM)
    WM_missing[:, -1, :] = True
    WM_missing[WM] = False
    WM_dist = edt.edt(np.logical_not(WM))
    WM_missing_dist = edt.edt(np.logical_not(WM_missing))
    WM_invalid = WM_missing_dist < WM_dist
    # CSF
    CSF = rim == 1
    # NOTE: this is very specific to the present dataset
    # Assuming that the top-most slice should be CSF
    CSF_missing = np.zeros_like(CSF)
    CSF_missing[:, 0, :] = True
    CSF_missing[CSF] = False
    CSF_dist = edt.edt(np.logical_not(CSF))
    CSF_missing_dist = edt.edt(np.logical_not(CSF_missing))
    CSF_invalid = CSF_missing_dist < CSF_dist
    # combine
    invalid = np.logical_or.reduce([missing, WM_invalid, CSF_invalid])
    metric_equidist[invalid] = 0

    return metric_equidist


def proc_LayNii(rim_fn, verbose=False):
    rim_fn_abs = os.path.abspath(rim_fn)
    if verbose:
        print('LayNii START')
    LayNii_folder = get_LayNii()
    # currently, layers are re-computed from the equidistant metric and equivol is not used
    cmd = f'{LayNii_folder}/LN2_LAYERS -rim {rim_fn_abs} -nr_layers 20 -equivol'
    process = subprocess.Popen(cmd, shell=True, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)
    out, err = process.communicate()
    errcode = process.returncode
    if errcode != 0:
        raise Exception(f'Error: {errcode}, {err}')
    if verbose:
        print('LayNii DONE')

    if rim_fn_abs.endswith('.nii'):
        base_fn = rim_fn_abs.split('.nii')[0]
    elif rim_fn_abs.endswith('.nii.gz'):
        base_fn = rim_fn_abs.split('.nii.gz')[0]
    else:
        raise Exception('Unknown file extension')
    
    metric_equidist_fn = base_fn + '_metric_equidist.nii'
    if os.path.isfile(metric_equidist_fn):
        metric_equidist = nb.load(metric_equidist_fn).get_fdata()
        return metric_equidist
    else:
        raise Exception(f'File {metric_equidist_fn} not found')
    

def seg_to_rim(seg, components={'GM': 0, 'WM': 1, 'Missing': 2, 'CSF': 3}):
    """
    """

    # color-code for laynii
    rim = np.copy(seg)
    rim[rim == components['CSF']] = 1.1
    rim[rim == components['GM']] = 3.1
    rim[rim == components['WM']] = 2.1
    rim[rim == components['Missing']] = 3.1  # treat as GM for LayNii
    rim = np.round(rim).astype(np.uint8)
    
    # add a surround
#    surround = np.ones(np.array(seg.shape) + [2, 2, 2]) * 4
#    surround[1:-1, 1:-1, 1:-1] = rim
#    rim = surround

    return rim


def proc_cortex_seg(data, downsampling, dilation=5, smoothing=5, components={0: 'GM', 1: 'WM', 4: 'CSF', 2: 'Missing'}):
    """Post-process a cortex segmentation.
    This involves:
    - downsampling (to isotropic resolution)
    - dilating the missing data component
    - smoothing the components

    Parameters
    ----------
    data : array
        The segmentation data
    downsampling : tuple or list
        The downsampling factors in each dimension
    dilation : int
        The diameter of the dilation ball
    smoothing : float
        The sigma of the gaussian smoothing
    components : dict
        The mapping of component values to component types
        e.g. {0: 'GM', 1: 'WM', 4: 'CSF', 2: 'Missing'}

    Returns
    -------
    processed_data : array
        The processed segmentation data

    """

    assert len(data.shape) == len(downsampling), 'Data and downsampling need to have the same number of dimensions'
    for d_i, d in enumerate(downsampling):
        slc = [slice(None, None, None) for _ in range(len(downsampling))]
        slc[d_i] = slice(None, None, d)
        slc = tuple(slc)
        data = data[slc]

    assert set(components.values()).issubset(set(['GM', 'WM', 'CSF', 'Missing'])), f'Need to provide components as {components.values()}'
    #components = [0, 1, 4, 2]  # GM, WM, CSF, Missing
    assert set(np.unique(data)).issubset(set(components)), f'The values in the segmentation are not {components}'
    comps = {}
    for value, component in components.items():
        comp = data == value
        comps[component] = comp

    # treat the missing data component conservaively
    # dilate it into the gray matter by 5 voxels
    comp = comps['Missing'].copy()
    comp = binary_dilation(comp, ball(dilation))
    comp = comp > comps['WM']
    comp = comp > comps['CSF']
    comps['Missing'] = comp

    # re-order
    comps = [comps[key] for key in ['GM', 'WM', 'Missing', 'CSF']]  # GM, WM, Missing, CSF

    for ic, component in enumerate(comps):
        comps[ic] = gaussian(component, (smoothing, smoothing, smoothing))
    
    processed_data = np.argmax(comps, axis=0)
    processed_data = processed_data.astype(data.dtype)

    return processed_data



def accept_labels(im, include):
    '''
    keep only the labels in include

    im : array
        label-image

    include : array-like
        global variable (not an argument), labels to keep

    TODO: make 'include' an argument
        Currently include is currently not passed as an argument (the global variable is used)
        It is possible to pass function arguments to da.map_blocks like da.mapblocks(function, a, b)
         but this seems to go wring when calling specific_array.map_blocks.
    '''
    mask = np.isin(im, include)
    im[np.logical_not(mask)] = 0
    return im


def discard_labels(im, include):
    '''
    discard the labels in include

    im : array
        label-image

    include : array-like
        global variable (not an argument), labels to keep

    TODO: make 'include' an argument
        Currently include is currently not passed as an argument (the global variable is used)
        It is possible to pass function arguments to da.map_blocks like da.mapblocks(function, a, b)
         but this seems to go wring when calling specific_array.map_blocks.
    '''
    mask = np.isin(im, include)
    im[mask] = 0
    return im


# chunked regionprops

# We can use regionprops_table to filter segmentations.
# For this, skimage.measure.regionprops_table can 1) take a while, and 2) needs all data in RAM.
#  e.g. 25min for a label image of shape [1000, 2000, 2000].
# It is much faster to apply regionprops_table on smaller chaunks and merge the results after.
#  e.g. 1 min for the same label image.
# These times might not generalize
# this is implemented here for
# properties=('label',
#            'area',
#            'bbox',
#            'area_bbox',
#            'extent',)
@dask.delayed
def my_regionprops_table(im, i, pos):
    # For speed and memory reasons, we want to apply regionprops_table on
    # chunks of a large zarr array.
    rp = skimage.measure.regionprops_table(im,
                                           properties=('label',
                                                       'area',
                                                       'bbox',
                                                       'area_bbox',
                                                       'extent',
                                                       'slice'))

    n = rp['label'].shape[0]
    rp['chunk_id'] = np.repeat(i, n)
    rp['chunk_x'] = np.repeat(pos[0], n)
    rp['chunk_y'] = np.repeat(pos[1], n)
    rp['chunk_z'] = np.repeat(pos[2], n)
    return pd.DataFrame(rp)


regionprops_meta = dd.utils.make_meta([('label', np.int64),
                                       ('area', np.float64),
                                       ('bbox-0', np.int64),
                                       ('bbox-1', np.int64),
                                       ('bbox-2', np.int64),
                                       ('bbox-3', np.int64),
                                       ('bbox-4', np.int64),
                                       ('bbox-5', np.int64),
                                       ('area_bbox', np.float64),
                                       ('extent', np.float64),
                                       ('slice', object),
                                       ('chunk_id', int),
                                       ('chunk_x', int),
                                       ('chunk_y', int),
                                       ('chunk_z', int), ])


def merge_regionprops(df):
    '''
    Note:
    to merge the 'slice' measure across chunks we simply select the first one
    '''
    i = len(df)
    area = df['area'].sum()
    x_start = df['x_start'].min()
    y_start = df['y_start'].min()
    z_start = df['z_start'].min()
    x_stop = df['x_stop'].max()
    y_stop = df['y_stop'].max()
    z_stop = df['z_stop'].max()
    area_bbox = (x_stop - x_start) * (y_stop - y_start) * (z_stop - z_start)
    extent = area / area_bbox
    return pd.DataFrame({'n_chunks': [i],
                         'area': [area],
                         'x_start': [x_start],
                         'y_start': [y_start],
                         'z_start': [z_start],
                         'x_stop': [x_stop],
                         'y_stop': [y_stop],
                         'z_stop': [z_stop],
                         'area_bbox': [area_bbox],
                         'extent': [extent],
                         'slice': [df.iloc[0]['slice']]})


def apply_chunked_regionprops(da_labels, meta=regionprops_meta):
    # prepare
    blocks = da_labels.to_delayed().ravel()
    regionprops = [dd.from_delayed(my_regionprops_table(data, i, data.key[1:]), meta=meta)
                   for i, data in enumerate(blocks)]
    regionprops = dd.concat(regionprops)

    # compute
    print('computing regionprops ...')
    ddf = regionprops.compute()
    print('cleaning regionprops ...')

    # post-processing
    # rename bbox-n into something more intuitive
    ddf = ddf.rename(columns={'bbox-0': 'x_start',
                              'bbox-1': 'y_start',
                              'bbox-2': 'z_start',
                              'bbox-3': 'x_stop',
                              'bbox-4': 'y_stop',
                              'bbox-5': 'z_stop'})

    # account for chunk positions
    ddf['x_start'] += ddf['chunk_x'] * da_labels.chunksize[0]
    ddf['y_start'] += ddf['chunk_y'] * da_labels.chunksize[1]
    ddf['z_start'] += ddf['chunk_z'] * da_labels.chunksize[2]

    ddf['x_stop'] += ddf['chunk_x'] * da_labels.chunksize[0]
    ddf['y_stop'] += ddf['chunk_y'] * da_labels.chunksize[1]
    ddf['z_stop'] += ddf['chunk_z'] * da_labels.chunksize[2]

    # drop columns that are not needed anymore
    # ddf = ddf.drop(columns=['chunk_x', 'chunk_y', 'chunk_z', 'chunk_id'])

    # labels found in multiple chunks
    u_label, n_chunks = np.unique(ddf['label'], return_counts=True)
    mask = n_chunks > 1
    multichunk_u_label = u_label[mask]
    multichunk_n_chunks = n_chunks[mask]
    multichunk_mask = ddf['label'].isin(multichunk_u_label)
    ddf_multichunk = ddf[multichunk_mask]

    # merge labels found in multiple chunks
    ddf_merged_regionprops = ddf_multichunk.groupby('label').apply(merge_regionprops)
    ddf_merged_regionprops = ddf_merged_regionprops.reset_index()
    ddf_merged_regionprops = ddf_merged_regionprops.drop(labels=['level_1'], axis=1)

    # labels contained within single chunks
    ddf_singlechunk = ddf.loc[np.logical_not(multichunk_mask), :]
    ddf_singlechunk = ddf_singlechunk.assign(n_chunks=1)

    # all labels
    ddf_merged = pd.concat([ddf_singlechunk, ddf_merged_regionprops])
    ddf_merged.sort_values(by='label', inplace=True)

    return ddf_merged


def get_zarr_size_on_disk(folder):
    '''
    Return size in GB
    '''
    n_bytes = sum(os.path.getsize(os.path.join(folder, f))
                  for f in os.listdir(folder)
                  if os.path.isfile(os.path.join(folder, f)))
    return n_bytes / 1024 ** 3


def medial_dist_transform(im):
    dist = distance_transform_edt(im).astype(np.float32)  # TODO: replace by seung labs' edt
    skel = skeletonize(im).astype(bool)
    out = np.zeros_like(dist)
    out[skel] = dist[skel]
    return out


def get_bins(minimum, maximum, step, check_uint16=True):
    """
    Get bins for digitization.
    Starting at 0, the bins are defined by the step size.
    The last bin includes the maximum value.
    If add_background is True, a background bin at zero is added.
    If check_uint16 is True, an exception is raised if the number of bins exceeds the maximum value of np.uint16.

    NOTE: use np.digitize with right=True such that the last bin is (data_max-data_step, data_max]

    Parameters
    ----------
    minimum : int
        The minimum value
    maximum : int
        The maximum value
    step : int
        The step size
    check_uint16 : bool
        Whether to check for uint16 overflow

    Returns
    -------
    bins : array
        The bins    
    """
    # bins, last bin including data_max
    # NOTE: use np.digitize with right=True such that the last bin is (data_max-data_step, data_max]
    bins = np.arange(minimum, maximum+step, step)
    if check_uint16:
        if len(bins)-1 > np.iinfo(np.uint16).max:
            raise Exception(f'Digitizing with N={len(bins)} bins will lead to {len(bins)-1} unique values: too many for cast to uint16')
    return bins


def get_layers(metric, n_depthbins=100, min_values_per_layer=100):
    """
    Get depthbins for digitization of metric into layers.

    """
    vmin = 0
    vmax = 1
    depthbins = np.linspace(vmin, vmax, n_depthbins + 1)
    hist, bin_edges = np.histogram(metric, bins=depthbins)
    if np.any(hist < min_values_per_layer):
        raise Exception(
            'Some layers do not have enough data - choose  fewer layers, (or a less restrictive criterion)!')
    # finally, sort from pia to WM (in typical LayNii matric space superficial depths are close to 1)
    # NOTE: cannot do this before, as np.histogram wants the bins in ascending order
    depthbins = depthbins[::-1]
    return depthbins


def indexed_hist(data, 
                 index, 
                 indexbins, 
                 mask=None, 
                 databins=None, 
                 mode=['count_foreground', 'get_foreground_values'][0], 
                 groupby_dim=None, 
                 threshold=.5, 
                 backbone = ['pd', 'np', 'hist'][2]):
    
    """Efficient computation of histograms in a 3D volume where a second 3D volume serves as 
    the coordinate system (index) that provides the basis for discretization into index-bins.
    This is useful for computing histograms of e.g. vessel density in a 3D volume, across layers of the cortex.
    Can also accept bins w.r.t the data (data-bins). 
    This is useful for looking at the distribution of e.g. vessel diameter within layer.
    Can also process a collection of 2D images that are concatenated along the first dimension (see groupby_dim).
    This is useful for processing 3D volumes in chunks (e.g. dask) where the 3D volume is split into 2D slices, 
    and allows e.g. post-hoc calculation of standard errors across slices.

    tldr. use 'hist' because it is fastest
    all three modes should give the same result (but this is not extensively tested)

    as the 'pd'-mode works quite differently - some work was needed to make it mirror the other two modes
    NOTE: when using backbone='pd'
    histogramming is done by digitizing and then counting unique values:
    np.digitize is run first, using databins as bins
    np.digitize applies bins as [0, 1), [1, 2), ..., [N-1, N)
    in addition, ONE more bin is added as needed: [-∞, 0) 
    in addition, the highest bin is extented as needed: [N-1, ∞]
    yielding up to N unique values/bins, from the input of N values (N-1 bins)
    the added 'zeroeth' bin returns 0 for data smaller than the first bin
    NOTE: when using backbone='np' or 'hist'
    histogramming is done directly with np.histogram or hist.numpy.histogram:
    bins are [a, b), ... and [y, z] for the last bin
    meaning that some data might not be counted if it falls outside the bins    
    HOWEVER, extra steps are implemented such that 'pd' mirrors the behavior of 'np' and 'hist' 
    this was done by 
    1) masking out the data that falls outside the bins, and 
    2) amending the results from np.digitize such that the last binnnnn is effectively [a, b] instead of [a, b)

    The (non-amended) difference is illustrated here:
    d = np.arange(-1, 6)
    print(d, len(d))
    bins = np.arange(5)
    print(bins, len(bins))
    digitized = np.digitize(d, bins)
    print(digitized, len(digitized))
    bins, edges = np.histogram(d, bins)
    print(bins, np.sum(bins))
    ...
    [-1  0  1  2  3  4  5] 7
    [0 1 2 3 4] 5       # 4 bins specified
    [0 1 2 3 4 5 5] 7   # digitization yields up to 6 unique values (i.e. 6 bins)
    [1 1 1 2] 5         # np.histogram only counts the data that falls within the 4 bins

    Parameters
    ----------
    data : array
        The data array
        Can be a masked array:
        If data is a masked array, the mask is applied before processing to BOTH data and index
        Only data INSIDE the mask is processed
        If data is a masked array, no mask should be provided
    index : array
        The index array
    indexbins : array
        The bins for the index
    mask : array, optional
        The mask array
        If provided, the data and index are masked before processing
        Only data INSIDE the mask is processed
        If data is a masked array, no mask should be provided
    databins : array
        The bins for the data
    groupby_dim : int, optional
        if provided, histograms are computed seperately for each slice along this dimension
        convenient for dask/chunked processing of 3D volumes
        also allows post-hoc calculation of e.g. standard-errors across 2D slices
    mode : str, optional
        The mode of operation
        'count_foreground' : count the number of foreground and background pixels where 
            backgound is defined by data<=threshold and foreground is defined by data>threshold
        'get_foreground_values' : count the number of pixels in each bin provided by databins
    threshold : float, optional
        The value that defines the background
    backbone : str, optional
        The backbone to use
        'pd' : use pandas
        'np' : use numpy
        'hist' : use hist (fastest!)

    Returns
    -------
    df : DataFrame
        The histogram
        with columns ['index', 'value', 'count']
        and optionally ['groupby_dim']
        where 'index' is the index-bin, 'value' is the data-bin, and 'count' is the number of pixels
    """

    if mode == 'get_foreground_values':
        if databins is None:
            raise Exception('databins need to be provided if mode is get_foreground_values')
    
    if groupby_dim is not None:
        if data.shape[groupby_dim] > np.iinfo(np.uint16).max:
            raise Exception(f'Casting to uint16 is not valid when creating a groupby_dim as long as {data.shape[groupby_dim]}')
        groupby_dim_index = np.arange(data.shape[groupby_dim])
        groupby_dim_bins = np.arange(data.shape[groupby_dim]+1)
        dim_id = [np.newaxis] * data.ndim
        dim_id[groupby_dim] = slice(None, None)
        groupby_dim_index = groupby_dim_index[tuple(dim_id)]
        reps = list(data.shape)
        reps[groupby_dim] = 1
        groupby_dim_index = np.tile(groupby_dim_index, tuple(reps)).ravel()

    if mask is not None:
        assert not isinstance(data, np.ma.masked_array), 'If data is a masked array, no mask should be provided'
        data = data[mask]
        index = index[mask]
        if groupby_dim is not None:
            groupby_dim_index = groupby_dim_index[mask]

    if isinstance(data, np.ma.masked_array):
        assert mask is None, 'If data is a masked array, no mask should be provided'
        mask = data.mask
        data = data[mask]
        index = index[mask]
        if groupby_dim is not None:
            groupby_dim_index = groupby_dim_index[mask]

#    if len(databins)-1 > np.iinfo(np.uint16).max:
#        raise Exception(f'Digitizing with N={len(bins)} bins will lead to {len(bins)-1} unique values: too many for cast to uint16')
#    data = np.digitize(data, databins, right=True).astype(np.uint16).ravel()
    data = data.ravel()
    #mask
#    if len(indexbins)-1 > np.iinfo(np.uint16).max:
#        raise Exception(f'Digitizing with N={len(bins)} bins will lead to {len(bins)-1} unique values: too many for cast to uint16')
#    idx = np.digitize(index, indexbins, right=True).astype(np.uint16).ravel()
#    # preserve making
#    idx = np.ma.array(idx, mask=np.isnan(index).ravel())
    idx = index
    idx = idx.ravel()

    if mode == 'count_foreground':
        # in this case, we only count 1) background (data<=threshold), and 2) foreground (data>threshold)
        if threshold is not None:
            data = (data>threshold).astype(np.uint8)

        if groupby_dim is None:
            
            if backbone == 'pd':
                if len(indexbins)-1 > np.iinfo(np.uint16).max:
                    raise Exception(f'Digitizing with N={len(bins)} bins will lead to {len(bins)-1} unique values: too many for cast to uint16')
                idx = np.digitize(index, indexbins).astype(np.uint16).ravel()
                # preserve making
                idx = np.ma.array(idx, mask=np.isnan(index).ravel())
                
                df = pd.DataFrame({'index': idx, 
                                   'data': data})#(data>threshold).astype(np.uint8)})
                df = df.groupby(['index'])['data'].value_counts().unstack().fillna(0)
                # rearrange
                df = df.reset_index()
                df = df.rename(columns={edge: 'count'+str(edge) for edge in np.arange(2)})
                df.columns.name = None
                df = df.reset_index(drop=True)
                df = pd.wide_to_long(df, ['count'], i=['index'], j='value')
                df = df.reset_index()
                
            else:
                # TODO should this rather not include databins
                in_edges = (indexbins)

                if backbone == 'np':
                    # if we are only interested in foreground vs. background
                    # two one-dimaensional histograms are faster
                    background, edges = np.histogram(idx[data==0], bins=indexbins)
                    foreground, edges = np.histogram(idx[data>0], bins=indexbins)
                    bins = np.hstack([background[:, np.newaxis], foreground[:, np.newaxis]])

                elif backbone == 'hist':
                    # if we are only interested in foreground vs. background
                    # two one-dimaensional histograms are faster
                    #print(np.unique(data))
                    a = idx[data==0]
                    a = a[np.isnan(a) == False]
                    background, edges = hist.numpy.histogram(a, bins=indexbins)
                    a = idx[data>0]
                    a = a[np.isnan(a) == False]
                    foreground, edges = hist.numpy.histogram(a, bins=indexbins)
#                    background, edges = hist.numpy.histogram(idx[np.logical_not(data)], bins=indexbins)
#                    foreground, edges = hist.numpy.histogram(idx[data], bins=indexbins)
#                    print(f'{np.sum(data==True)}_{np.nansum(idx[data==True])}')
                    bins = np.hstack([background[:, np.newaxis], foreground[:, np.newaxis]])
                    
                bin_idx = np.meshgrid(*[np.arange(dim) for dim in bins.shape], indexing='ij')
                df = pd.DataFrame({'index': bin_idx[0].ravel(), 
                                   'value': bin_idx[1].ravel(), 
                                   'count': bins.ravel()})

        else:
            
            if backbone == 'pd':
                
                if len(indexbins)-1 > np.iinfo(np.uint16).max:
                    raise Exception(f'Digitizing with N={len(bins)} bins will lead to {len(bins)-1} unique values: too many for cast to uint16')
                idx = np.digitize(index, indexbins).astype(np.uint16).ravel()
                # preserve making
                idx = np.ma.array(idx, mask=np.isnan(index).ravel())
                
                df = pd.DataFrame({'index': idx, 
                                   'groupby_dim': groupby_dim_index, 
                                   'data': data})#(data>background_val).astype(np.uint8)})
                df = df.groupby(['index', 'groupby_dim'])['data'].value_counts().unstack().fillna(0)
                # rearrange
                df = df.reset_index()
                df = df.rename(columns={edge: 'count'+str(edge) for edge in np.arange(2)})
                df.columns.name = None
                df = df.reset_index(drop=True)
                df = pd.wide_to_long(df, ['count'], i=['index', 'groupby_dim'], j='value')
                df = df.reset_index()

            else:
                comb_idx = np.hstack([idx[:, np.newaxis], 
                                      groupby_dim_index[:, np.newaxis], 
                                      data[:, np.newaxis]])
                in_edges = (indexbins, groupby_dim_bins, databins)

                if backbone == 'np':
                    # TODO could speed this up by splitting into two 2d histograms (one foreground, one background, see above)
                    bins, edges = np.histogramdd(comb_idx, bins=in_edges)

                elif backbone == 'hist':
                    # TODO could speed this up by splitting into two 2d histograms (one foreground, one background, see above)
                    bins, edges = hist.numpy.histogramdd(comb_idx, bins=in_edges)
                
                bin_idx = np.meshgrid(*[np.arange(dim) for dim in bins.shape], indexing='ij')
                df = pd.DataFrame({'index': bin_idx[0].ravel(), 
                                   'groupby_dim_index': bin_idx[1].ravel(), 
                                   'value': bin_idx[2].ravel(), 
                                   'count': bins.ravel()})
        
    elif mode == 'get_foreground_values':
        # in this case, we count in databins
        if groupby_dim is None:
            
            if backbone == 'pd':
                if len(indexbins)-1 > np.iinfo(np.uint16).max:
                    raise Exception(f'Digitizing with N={len(bins)} bins will lead to {len(bins)-1} unique values: too many for cast to uint16')
                
                # mirror np.histogram behaviour:
                mask = np.logical_or(index < np.min(indexbins), index > np.max(indexbins))
                b_mask = index == np.max(indexbins)
                idx = np.digitize(index, indexbins).astype(np.uint16).ravel()
                # for the final bin (which is [a, b])
                # by setting b to a
                idx[b_mask] = len(indexbins) - 1
                # mirror np.histogram behaviour:
                # mask out data outside of the bins
                idx = np.ma.masked_array(idx, mask)
                # preserve making
                idx.mask = np.logical_or(idx.mask, np.isnan(index).ravel())
                idx = np.ma.array(idx, mask=np.isnan(index).ravel())

                if len(databins)-2 > np.iinfo(np.uint16).max:
                    raise Exception(f'Digitizing with N={len(bins)} bins will lead to {len(bins)-1} unique values: too many for cast to uint16')
                
                # mirror np.histogram behaviour:
                mask = np.logical_or(data < np.min(databins), data > np.max(databins))
                b_mask = data == np.max(databins)
                data = np.digitize(data, databins).astype(np.uint16).ravel()
                # for the final bin (which is [a, b])
                # by setting b to a
                data[b_mask] = len(databins) - 1
                # mirror np.histogram behaviour:
                # mask out data outside of the bins
                data = np.ma.masked_array(data, mask)

                df = pd.DataFrame({'index': idx, 
                                   'data': data})
                df = df.groupby(['index'])['data'].value_counts().unstack().fillna(0)
                # rearrange
                df = df.reset_index()
                df = df.rename(columns={edge: 'count'+str(edge) for edge in np.arange(len(indexbins)-1)})
                df.columns.name = None
                df = df.reset_index(drop=True)
                df = pd.wide_to_long(df, ['count'], i=['index'], j='value')
                df = df.reset_index()
                # mirror np.histogram behaviour:
                # start bins at 0
                df.loc[:, 'value'] = df['value'] - 1
                df.loc[:, 'index'] = df['index'] - 1
                
            else:
                # NOTE mod
                comb_idx = np.hstack([idx[:, np.newaxis],
                                      data[:, np.newaxis]])
                
                in_edges = (indexbins, databins)
                if backbone == 'np':
                    bins, edges = np.histogramdd(comb_idx, bins=in_edges)
                elif backbone == 'hist':
                    bins, edges = hist.numpy.histogramdd(comb_idx, bins=in_edges)
                    
                bin_idx = np.meshgrid(*[np.arange(dim) for dim in bins.shape], indexing='ij')
                df = pd.DataFrame({'index': bin_idx[0].ravel(), 
                                   'value': bin_idx[1].ravel(), 
                                   'count': bins.ravel()})
        else:
            if backbone == 'pd':
                if len(indexbins)-1 > np.iinfo(np.uint16).max:
                    raise Exception(f'Digitizing with N={len(bins)} bins will lead to {len(bins)-1} unique values: too many for cast to uint16')
                
                # mirror np.histogram behaviour:
                mask = np.logical_or(index < np.min(indexbins), index > np.max(indexbins))
                b_mask = index == np.max(indexbins)
                idx = np.digitize(index, indexbins).astype(np.uint16).ravel()
                # for the final bin (which is [a, b])
                # by setting b to a
                idx[b_mask] = len(indexbins) - 1
                # mirror np.histogram behaviour:
                # mask out data outside of the bins
                idx = np.ma.masked_array(idx, mask)
                # preserve making
                idx.mask = np.logical_or(idx.mask, np.isnan(index).ravel())
                idx = np.ma.array(idx, mask=np.isnan(index).ravel())

                if len(databins)-2 > np.iinfo(np.uint16).max:
                    raise Exception(f'Digitizing with N={len(bins)} bins will lead to {len(bins)-1} unique values: too many for cast to uint16')
                
                # mirror np.histogram behaviour:
                mask = np.logical_or(data < np.min(databins), data > np.max(databins))
                b_mask = data == np.max(databins)
                data = np.digitize(data, databins).astype(np.uint16).ravel()
                # for the final bin (which is [a, b])
                # by setting b to a
                data[b_mask] = len(databins) - 1
                # mirror np.histogram behaviour:
                # mask out data outside of the bins
                data = np.ma.masked_array(data, mask)

                df = pd.DataFrame({'index': idx, 
                                   'groupby_dim': groupby_dim_index, 
                                   'data': data})
                df = df.groupby(['index', 'groupby_dim'])['data'].value_counts().unstack().fillna(0)
                # rearrange
                df = df.reset_index()
                # TODO this: np.arange(len(indexbins)-1)
                # should be: np.arange(len(databins))
                df = df.rename(columns={edge: 'count'+str(edge) for edge in np.arange(len(indexbins)-1)})
                df.columns.name = None
                df = df.reset_index(drop=True)
                df = pd.wide_to_long(df, ['count'], i=['index', 'groupby_dim'], j='value')
                df = df.reset_index()
                
            else:
                comb_idx = np.hstack([idx[:, np.newaxis], 
                                      groupby_dim_index[:, np.newaxis], 
                                      data[:, np.newaxis]])
                in_edges = (indexbins, groupby_dim_bins, databins)
                if backbone == 'np':
                    bins, edges = np.histogramdd(comb_idx, bins=in_edges)

                elif backbone == 'hist':
                    bins, edges = hist.numpy.histogramdd(comb_idx, bins=in_edges)
                bin_idx = np.meshgrid(*[np.arange(dim) for dim in bins.shape], indexing='ij')
                df = pd.DataFrame({'index': bin_idx[0].ravel(), 
                                   'groupby_dim_index': bin_idx[1].ravel(), 
                                   'value': bin_idx[2].ravel(), 
                                   'count': bins.ravel()})        

    else:
        raise NotImplementedError(f'mode {mode} is not implemented')

    return df


def old_indexed_hist(index, 
                     data, 
                     depthbins, 
                     mode=['count_foreground', 'get_foreground_values'][0], 
                     each_layer_needs_data=False):
    """Efficient computation of histograms in a 3D volume where a second 3D volume serves as 
    the coordinate system (index) that provides the basis for discretization into bins.
    Can also process a collection of 2D images that are concatenated along the first dimension (see zpos_per_layer).

    E.g. 
        Give me 
            an index (a 3D volume, e.g. an distance map), 
            data (a 3D volume, e.g. intensity measurements), and 
            depthbins (the bin edges w.r.t. the index) 
        ... and get back the ...
            intensity measurements per bin
        or
            the number of foreground and background voxels per bin

    Some algorithmic details:
        100 layers can be described with 7 splits
        each having the potential to seperate the data into batches
        of 2*n layers where n is
        [64 32 16 8 4 2 1]

        Layer 64 has binary representation [1 0 0 0 0 0 0],
        meaning that we can split the data 7 times
        (in 2 halves each time, assuming that the data is uniformly distributed)

        Given 100 depthbins, defined by 101 edges,
        the boundaries of the above 7 splits are given by:
        lower histogram-edges [64 64 64 64 64 64 64], and
        upper histogram-edges [100 96 80 72 68 66 65]
        (which provide indices into the depthbins)

        While splitting the data 7 times to get a single layer does not seem efficient,
        it is efficient when we have to split the data for many layers.
        e.g. the first split can be memorized and reused for 100-64=36 layers.
        while the 7th split can be used for 2 layers.
        More importantly, while we split the data quite often, we do so on sucessive smaller parts.

    Parameters
    ----------
    index : np.array
        3D array of indices
    data : np.array
        3D array of data
    depthbins : np.array
        array of bin edges
    mode : str
        if 'count_foreground',
            data should be a segmentation (binary, foreground vs background), and
            foreground_per_layer is the number of foreground voxels
        if 'get_foreground_values',
            data should be something like a distance map (of a segmentation)
            (where background voxels are 0 and foreground voxels are >0), and
            values_per_layer is a list of arrays containing the values within layers
            (which need to be binned/counted in a separate step)
    each_layer_needs_data : bool
        if True, an exception is raised if a bin does not have any data

    Returns
    -------
    if mode == 'count_foreground',
        layer : np.array
        voxels_per_layer : np.array
        foreground_per_layer : np.array
        zpos_per_layer : np.array

    
    if mode == 'get_foreground_values',
        layer : np.array
        voxels_per_layer : np.array
        values_per_layer : np.array
        zpos_per_layer : np.array


    Note
    ----
    In combination with dask, it can be used for efficient and BATCHED computation of histograms over large 3D volumes.

    Generally, a naive, iterative implementation of histogramming would be slow:
    In particular, iterating over bins would imply that the full data has to me indexed again and again (once per bin).
    Here, we avoid this by a binary search where the data is successively split into two smaller parts.
    This is done as often as required such that the smallest parts only contain 2 layers.
    Then each layer can be identified by its binary representation.
    E.g. With 4 layers, we can split the data twice:
    First in two halves, and each of those into 2 quaters.
    Then, layer 1 is identified by splits [00]:
    i.e. first half, first quater
    Crucially, by memorizing earlier splits, we only have to search 1x100%, and 2x50% of the data (instead of 4x100%).

    TODO: Is there a faster way?
    Tried sorting but it seems slower (or at least not faster)
    e.g. the initial steps already take roughly the same time as the whole binary search
        data_flat = data.ravel()
        argsort = np.argsort(index_sorted)
        data_sorted = data_flat[argsort]
        index_sorted = index.ravel()[argsort]
    Maybe there are still circumstances where sorting is faster? (depending on data size or number of depthbins?!)
    Especially np.searchsorted would be fast for finding bin edges in a sorted array
    """
    if not np.all(np.diff(depthbins) > 0):
        raise Exception('depthbins need to be strictly increasing')        

    # depth in the 3D volume, assuming that the first dimension is the z-dimension
    # useful when processing a collection of 2D images that are concatenated along the first dimension
    # then, the returned zpos_per_layer allows to disambiguate the 2D images
    depth = np.tile(np.arange(index.shape[0])[..., None, None], (1, index.shape[1], index.shape[2]))
    
    # number of splits required
    n_depthbins = len(depthbins)-1
    n_splits = int(np.ceil(math.log(n_depthbins, 2)))
    
    # layers
    bins = np.arange(n_depthbins)  # [0, 1, 2, ... ]
    
    # describe histogram-bins by successive splits
    # e.g. [1, 0, 0, 1] is the layer you get when
    # 1st, taking the second half, 
    # 2nd, taking the first half of that
    # 3rd, taking the first half of that
    # 4th, taking the second half of that
    bin_split_repr = [list(np.binary_repr(bin_i, width=n_splits)) for bin_i in bins]
    bin_split_repr = np.array(bin_split_repr).astype(int)
    
    # Describe histogram-bins by histogram-edges (lower and upper) of their sucessive splits
    # e.g. 
    # 100 layers can be described with 7 splits
    # each having the potential to seperate the data into batches 
    # of 2*n layers where n is 
    # [64 32 16 8 4 2 1]
    #
    # Layer 64 has binary representation [1 0 0 0 0 0 0], 
    # meaning that we can split the data 7 times 
    # (in 2 halves each time, assuming that the data is uniformly distributed)
    #
    # Given 100 depthbins, defined by 101 edges, 
    # the boundaries of the above 7 splits are given by:
    # lower histogram-edges [64 64 64 64 64 64 64], and
    # upper histogram-edges [100 96 80 72 68 66 65]
    # (which provide indices into the depthbins)
    #
    # While splitting the data 7 times to get a single layer does not seem efficient, 
    # it is efficient when we have to split the data for many layers.
    # e.g. the first split can be memorized and reused for 100-64=36 layers.
    # while the 7th split can be used for 2 layers.
    # More importantly, while we split the data quite often, we do so on sucessively smaller parts.

    # lower histogram-edges
    bsr_lower_edge = np.zeros_like(bin_split_repr)
    for i, bsr_i in enumerate(bin_split_repr):
        ind = 2**np.where(bsr_i[::-1])[0]
        bsr_lower_edge[i][bsr_i>0] = ind[::-1]
    bsr_lower_edge = np.cumsum(bsr_lower_edge, axis=1)
    
    # upper histogram-edges
    bsr_upper_edge = np.zeros_like(bin_split_repr)
    for j, bj in enumerate(bsr_lower_edge.T):
        un = np.unique(bj)
        for i, u in enumerate(un[:-1]):
            bsr_upper_edge[:, j][bj==u] = un[i+1]
        bsr_upper_edge[:, j][bj==un[-1]] = n_depthbins 

    # initialize output
    if mode == 'count_foreground':
        voxels_per_layer = np.zeros(n_depthbins)
        zpos_per_layer = [[] for i in range(n_depthbins)]
        foreground_per_layer = np.zeros(n_depthbins)
    elif mode == 'get_foreground_values':
        voxels_per_layer = [[] for i in range(n_depthbins)]
        values_per_layer = [[] for i in range(n_depthbins)]
        zpos_per_layer = [[] for i in range(n_depthbins)]
        foreground_per_layer = np.zeros(n_depthbins)
    else:
        raise NotImplementedError(f'mode {mode} not implemented')

    previous = None
    # keep the current index and data sub-arrays in memory
    mask_halfs = []
    data_halfs = []
    depth_halfs = []
    # for each histogram-bin
    for li, (bl, bu) in enumerate(zip(bsr_lower_edge, bsr_upper_edge)):
        # keep track of the histogram-edges of the current sub-arrays
        this = []
        for h, (l, u) in enumerate(zip(bl, bu)):
            this.append([depthbins[l], depthbins[u]])
        # from the second layer onwards ...
        if previous:
            # for each successive split
            for h, (l, u) in enumerate(this):
                # if the necessary split has already been computed, we can skip forward
                if previous[h] == [l, u]:
                    pass
                else:
                    if h == 0:
                        mask_half = np.logical_and(index > l, 
                                                   index <= u)
                        mask_halfs[h] = index[mask_half]
                        data_half = data[mask_half]
                        data_halfs[h] = data_half
                        depth_half = depth[mask_half]
                        depth_halfs[h] = depth_half
                    else:
                        mask_half = np.logical_and(mask_halfs[h-1] > l, 
                                                   mask_halfs[h-1] <= u)
                        if each_layer_needs_data:
                            if not np.any(mask_half):
                                raise Exception('Layer ', str(li), ' does not have data')
                        mask_halfs[h] = mask_halfs[h-1][mask_half]
                        data_half = data_halfs[h-1][mask_half]
                        data_halfs[h] = data_half
                        depth_half = depth_halfs[h-1][mask_half]
                        depth_halfs[h] = depth_half
        
        # first layer, initialize
        else:
            for h, (l, u) in enumerate(this):
                if h == 0:
                    mask_half = np.logical_and(index > l, 
                                               index <= u)
                    mask_halfs.append(index[mask_half])
                    data_half = data[mask_half]
                    data_halfs.append(data_half)
                    depth_half = depth[mask_half]
                    depth_halfs.append(depth_half)
                else:
                    mask_half = np.logical_and(mask_halfs[h-1] > l, 
                                               mask_halfs[h-1] <= u)
                    mask_halfs.append(mask_halfs[h-1][mask_half])
                    data_half = data_halfs[h-1][mask_half]
                    data_halfs.append(data_half)
                    depth_half = depth_halfs[h-1][mask_half]
                    depth_halfs.append(depth_half)

        # get the 'volume' and density for each layer
        voxels_per_layer[li] = mask_halfs[-1].size
        if mode == 'count_foreground':
            if voxels_per_layer[li]:
                # get z-position of foreground voxels
                #zpos_per_layer[li] = depth_halfs[-1][data_halfs[-1] > 0]
                # sum the foreground voxels
                #foreground_per_layer[li] = np.sum(data_halfs[-1])

                df = pd

            else:
                zpos_per_layer[li] = np.array([0])
                foreground_per_layer[li] = 0
        elif mode == 'get_foreground_values':
            if voxels_per_layer[li]:
                # get foreground voxel values
                values_per_layer[li] = data_halfs[-1][data_halfs[-1] > 0]
                # get z-position of foreground voxels
                zpos_per_layer[li] = depth_halfs[-1][data_halfs[-1] > 0]
                # sum the foreground voxels
                foreground_per_layer[li] = values_per_layer[li].size
            else:
                values_per_layer[li] = np.array([0])
                zpos_per_layer[li] = np.array([0])
                foreground_per_layer[li] = 0
        # keep track of the histogram-edges of the previous sub-arrays
        previous = copy.deepcopy(this)
#        print([dh.shape for dh in data_halfs])
        
    if mode == 'count_foreground':
        layer = 1
        #layer = np.hstack([np.ones(len(valpl))*i for i, valpl in enumerate(foreground_per_layer)])
        #voxels_per_layer = np.hstack([np.ones(len(valpl))*voxpl for valpl, voxpl in zip(foreground_per_layer, voxels_per_layer)])
        #foreground_per_layer = np.hstack(foreground_per_layer)
        zpos_per_layer = np.hstack(zpos_per_layer)
        return layer, voxels_per_layer, foreground_per_layer, zpos_per_layer
    elif mode == 'get_foreground_values':
        layer = np.hstack([np.ones(len(valpl))*i for i, valpl in enumerate(values_per_layer)])
        voxels_per_layer = np.hstack([np.ones(len(valpl))*voxpl for valpl, voxpl in zip(values_per_layer, voxels_per_layer)])
        values_per_layer = np.hstack(values_per_layer)
        zpos_per_layer = np.hstack(zpos_per_layer)
        return layer, voxels_per_layer, values_per_layer, zpos_per_layer
    
@dask.delayed
def my_indexed_hist_ori(i, index, ori, hsv, depthbins, wrap, block_info=None):
    """see my_indexed_hist
    The difference is that my_indexed_hist counts the number of foreground voxels while 
    my_indexed_hist_ori counts the number of voxels for each unique radius (at a given numerical precision: np.round(data, 3))
    """
    metric = index
    
    data = hsv[..., 0]

    if wrap:
        data = .5 - np.abs(data - .5) # NOTE wrap orientations from vertical (0) to horizontal (.5)
    data = np.ma.masked_array(data, mask=np.all(ori == 0, axis=-1))

#    coord = metric
#    z, x, y = coord.shape
#    coord_corr = np.arctan(zoom(np.diff(coord, 1, axis=0), (x/(x-1), y/y)) / 
#                           zoom(np.diff(coord, 1, axis=1), (x/x, y/(y-1))))
#    coord_corr = (np.pi/2 - coord_corr) / np.pi
#    data[np.logical_not(data.mask)] -= coord_corr[np.logical_not(data.mask)]
#    data[data>1] = data[data>1] - 1
#    data[data<0] = data[data<0] + 1

    data = np.round(data, 3) # TODO parameterize
    
    layer, voxels_per_layer, values_per_layer = indexed_hist(metric, 
                                                             data, 
                                                             depthbins, 
                                                             binary_data = False)
    # there are many more centerline-voxels than different radii
    # so we can summarize by counting the number of voxels for each unique radius
    t = pd.DataFrame({'layer': layer, 
                      'voxels_per_layer': voxels_per_layer, 
                      'values_per_layer': values_per_layer})
    
    def count_radii(voxels_of_layer, values_of_layer):
        radii, counts = np.unique(values_of_layer, return_counts=True)
        return pd.DataFrame({'voxels_per_layer': np.repeat(voxels_of_layer.iloc[0], len(radii)), 
                             'radius': radii, 
                             'count': counts})

    radius_counts = t.groupby('layer').apply(lambda x: 
                                             count_radii(x['voxels_per_layer'], 
                                                         x['values_per_layer']))
    radius_counts = radius_counts.reset_index()
    radius_counts = radius_counts.drop(labels=['level_1'], axis=1)
    
    radius_counts['chunk_id'] = np.repeat(i, len(radius_counts)).astype(np.uint32)
    
    radius_counts['layer'] = radius_counts['layer'].astype(np.uint16)
    radius_counts['voxels_per_layer'] = radius_counts['voxels_per_layer'].astype(np.uint64)
    radius_counts['radius'] = radius_counts['radius'].astype(np.float32)
    radius_counts['count'] = radius_counts['count'].astype(np.uint64)

    #print(block_info[0])
    #chunk_id = np.ravel_multi_index(block_info[0]['chunk-location'], 
    #                                combined.numblocks)
    #chunk_id = np.ones(len(layer))*chunk_id
    #out = np.vstack([chunk_id, layer, voxels_per_layer, values_per_layer])
    
#    return layer, voxels_per_layer, values_per_layer
#    return pd.DataFrame({'chunk_id': int(i),
#                         'layer': layer, 
#                         'size': voxels_per_layer, 
#                         'radius': values_per_layer})
    return radius_counts


@dask.delayed
def delayed_indexed_hist(i, data, index, indexbins, binary_data=True, databins=None, 
                         background_val=None, groupby_dim=None, block_info=None):
    """
    A wrapper for indexed_hist.
    It is used to provide a dask.delayed object that can be used in a dask.compute call.
    Also adds the chunk_id to the output.


    Parameters
    ----------
    i : int
        The chunk id
    data : np.array
        The data array
        Can be a masked array (see indexed_hist)
    index : np.array
        The index array
    indexbins : np.array
        The bins for the index
    binary_data : bool
        If True, the data is treated as binary (foreground vs. background)
    databins : np.array
        The bins for the data
    background_val : int
        used when binary_data is True
        The value that defines the background
        i.e. background is data<=background_val, foreground is data>background_val
        
        TODO: could apply this to continuous data as well
        e.g. could add an additional bin for background
        by asserting that background_val is already in databins (making sure that other bins do not get 'split')
        and by adding background_val a second time to databins? does [background_val, background_val) work?
    block_info : dict
        The block info 

    Returns
    -------
    pd.DataFrame
        The histogram
    """
    if binary_data:
        if databins is not None:
            raise ValueError('databins should not be provided if binary_data is True')
        #layer, voxels_per_layer, foreground_per_layer = indexed_hist(index,
        #                                                             data,
        #                                                             indexbins,
        #                                                             mode='count_foreground', 
        #                                                             threshold=background_val)
        # TODO return layer
        #return pd.DataFrame({'chunk_id': int(i),
        #                     'layer_volume': voxels_per_layer,
        #                     'vessel_volume': foreground_per_layer})
        df = indexed_hist(data, 
                          index,
                          indexbins,
                          mode='count_foreground', 
                          threshold=background_val, 
                          groupby_dim=groupby_dim)
        
    else:
        # TODO rename radius
        #layer, voxels_per_layer, values_per_layer = indexed_hist(index,
        #                                                         data,
        #                                                         indexbins,
        #                                                         databins=databins,
        #                                                         mode='get_foreground_values')
        # there are many more centerline-voxels than different radii
        # so we can summarize by counting the number of voxels for each unique radius
        #df = pd.DataFrame({'layer': layer,
        #                  'voxels_per_layer': voxels_per_layer,
        #                  'values_per_layer': values_per_layer})
        df = indexed_hist(data, 
                          index,
                          indexbins,
                          databins=databins,
                          mode='get_foreground_values', 
                          groupby_dim=groupby_dim)

        #def count_radii(voxels_of_layer, values_of_layer):
        #    radii, counts = np.unique(values_of_layer, return_counts=True)
        #    return pd.DataFrame({'voxels_per_layer': np.repeat(voxels_of_layer.iloc[0], len(radii)),
        #                         'radius': radii,
        #                         'count': counts})
        #radius_counts = t.groupby('layer').apply(lambda x:
        #                                        count_radii(x['voxels_per_layer'],
        #                                                    x['values_per_layer']))
        #radius_counts = radius_counts.reset_index()
        #radius_counts = radius_counts.drop(labels=['level_1'], axis=1)
        #radius_counts['chunk_id'] = np.repeat(i, len(radius_counts)).astype(np.uint32)
        #radius_counts['layer'] = radius_counts['layer'].astype(np.uint16)
        #radius_counts['voxels_per_layer'] = radius_counts['voxels_per_layer'].astype(np.uint64)
        #radius_counts['radius'] = radius_counts['radius'].astype(np.float32)
        #radius_counts['count'] = radius_counts['count'].astype(np.uint64)

    df['chunk_id'] = np.repeat(i, len(df))
    return df


# TODO name binary or continuous
# delayed_indexed_hist
delayed_indexed_hist_gd_meta = dd.utils.make_meta([('index', np.int64),
                                                     ('value', np.float16),
                                                     ('count', np.int64), 
                                                     ('groupby_dim', int),
                                                     ('chunk_id', int)])

delayed_indexed_hist_meta = dd.utils.make_meta([('index', int),
                                                ('value', np.float16),
                                                ('count', int), 
                                                ('chunk_id', int)])





@dask.delayed
def my_indexed_hist_radii(i, index, data, block_info=None):
    metric = index
    data = data
    data = medial_dist_transform(data)
    layer, voxels_per_layer, values_per_layer = indexed_hist(metric,
                                                             data,
                                                             binary_data=False)
    # there are many more centerline-voxels than different radii
    # so we can summarize by counting the number of voxels for each unique radius
    t = pd.DataFrame({'layer': layer,
                      'voxels_per_layer': voxels_per_layer,
                      'values_per_layer': values_per_layer})

    def count_radii(voxels_of_layer, values_of_layer):
        radii, counts = np.unique(values_of_layer, return_counts=True)
        return pd.DataFrame({'voxels_per_layer': np.repeat(voxels_of_layer.iloc[0], len(radii)),
                             'radius': radii,
                             'count': counts})

    radius_counts = t.groupby('layer').apply(lambda x:
                                             count_radii(x['voxels_per_layer'],
                                                         x['values_per_layer']))
    radius_counts = radius_counts.reset_index()
    radius_counts = radius_counts.drop(labels=['level_1'], axis=1)

    radius_counts['chunk_id'] = np.repeat(i, len(radius_counts)).astype(np.uint32)

    radius_counts['layer'] = radius_counts['layer'].astype(np.uint16)
    radius_counts['voxels_per_layer'] = radius_counts['voxels_per_layer'].astype(np.uint64)
    radius_counts['radius'] = radius_counts['radius'].astype(np.float32)
    radius_counts['count'] = radius_counts['count'].astype(np.uint64)

    # print(block_info[0])
    # chunk_id = np.ravel_multi_index(block_info[0]['chunk-location'],
    #                                combined.numblocks)
    # chunk_id = np.ones(len(layer))*chunk_id
    # out = np.vstack([chunk_id, layer, voxels_per_layer, values_per_layer])

    #    return layer, voxels_per_layer, values_per_layer
    #    return pd.DataFrame({'chunk_id': int(i),
    #                         'layer': layer,
    #                         'size': voxels_per_layer,
    #                         'radius': values_per_layer})
    return radius_counts


# TODO consider if smaller datatypes are sufficient
my_indexed_hist_radii_meta = dd.utils.make_meta([('layer', np.uint16),
                                                 ('voxels_per_layer', np.uint64),
                                                 ('radius', np.float16),
                                                 ('count', np.uint64),
                                                 ('chunk_id', np.uint32)])