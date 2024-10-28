import sys
import os
import copy

import subprocess

import numpy as np
import tifffile
from tqdm import tqdm

import nibabel as nb
import skimage.io as skio

import dask
import dask.array as da
from dask import delayed

import zarr


def lazy_chunked_IP(im_zarr, axis=0, mode='max'):
    """Compute a maximum- or minimum- intensity projection of a dask array in chunks.
    TODO: could
    """
    
    mode = mode.lower()
    if mode == 'min':
        fun = np.min
    elif mode == 'max':
        fun = np.max
    elif mode == 'mean':
        fun = np.mean
    elif mode == 'sum':
        fun = np.sum
    else:
        raise NotImplementedError(f'mode {mode} not implemented')
    
    dtype = im_zarr.dtype
    slc = tuple([slice(0, 1) for dim in range(im_zarr.ndim)])
    shape = fun(im_zarr.blocks[slc].compute(), axis=axis, keepdims=True).shape
    
    def chunk_mip(im, axis=axis, keepdims=True):
        return fun(im, axis=axis, keepdims=keepdims)
    
    mip = im_zarr.map_blocks(chunk_mip, dtype=dtype, chunks=shape)
    mip = fun(mip, axis=axis, keepdims=True)
    return mip

def get_LayNii():
    """Download and unzip LayNii"""
    LayNii_url = 'https://github.com/layerfMRI/LAYNII/releases/download/v2.7.0/LayNii_v2.7.0_Linux64.zip'
    if not os.path.isdir('LayNii_v2.7.0'):
        subprocess.run(['wget', 
                        LayNii_url])
        subprocess.run(['unzip', 'LayNii_v2.7.0_Linux64.zip'])
    return 'LayNii_v2.7.0'
        

def from_file(fn):
    """
    """
    if fn.lower().endswith('.tif') or fn.lower().endswith('.tiff'):
        try:
            return tifffile.memmap(fn)
        except:
            return tifffile.imread(fn)
    elif fn.lower().endswith('.nii') or fn.lower().endswith('.nii.gz'):
        return nb.load(fn).get_fdata()
    elif fn.lower().endswith('.zarr'):
        return da.from_zarr(fn)

def to_nii(data, fn, postfix, affine=np.eye(3), overwrite=False):
    """Save a numpy array as nifti file.

    Parameters
    ----------
    data : np.array
        data to save
    fn : str
        path to save the data
    postfix : str
        postfix for the filename
    affine : np.array
        affine transformation
    overwrite : bool
        overwrite existing file

    Returns
    -------
    out_fn : str
        path to the saved file
    """
    if not postfix.endswith('.nii'):
        postfix += '.nii'
    out_fn = fn + postfix
    
    if not overwrite:
        if os.path.isfile(out_fn):
            return out_fn
            
    nii_im = nb.Nifti1Image(data, np.eye(4))
    nb.save(nii_im, out_fn)
    return out_fn

def mip_stride(data, mip_r, mip_step, stride, postprocess=None, verbose=False):
    """Quick-and-dirty downsampling
    First, downsample along the first dimension by maximum-intensity projections
    Second, downsample along the 2nd+3rd dimension by striding
    Third, (optionally) postprocess

    Parameters
    ----------
    data : array-like
        data to downsample
    mip_r : int
        radius for maximum-intensity projection
        the projection then happens over [-mip_r, mip_r+1]
    mip_step : int
        step size for maximum-intensity projection
        starting at floor(z/2 % mip_step) where z is the length of the first dimension
    stride : int
        step size for striding
    postprocess : function or None
        function to postprocess the downsampled data
    verbose : bool
        print information about the downsampling

    Returns
    -------
    im : np.array
        downsampled data

    Notes
    -----
    - the start/end of the maximum-intensity projection is such 
    that the center of the window fits into the array 
    (but not necessariely its full width)
    - this is probably more valid for zooming of the results, 
    but might leduce contrast near the edges
    """
    z = data.shape[0]
    # assure symmetric placement of windows (up to integer rounding)
    start = int(np.floor((z/2) % mip_step))
    s = np.arange(start, z, mip_step)
    im = []

    dask_input = isinstance(data, da.Array)
    
    for i, s_i in enumerate(s):
        im_i = data[np.max([0, s_i-mip_r]):np.min([z, s_i+mip_r+1]), ...]
        im_i = np.max(im_i, axis=0)
        im.append(im_i[::stride, ::stride])
        
    if dask_input:
        im = [im_i.compute() for im_i in im]

    if postprocess is not None:
        im = [postprocess(im_i) for im_i in im]
        
    im = np.stack(im)
    
    if verbose:
        print(f'mip_stride downsampling from {data.nbytes / (1024**3)} GB, shape:{data.shape} to {im.nbytes / (1024**3)} GB, shape:{im.shape}')
    return im
    

def get_sub_volumes_for_manual_annotation(data, out_folder, out_prefix, w, Z, Y, X):
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    for z in Z:
        for y in Y:
            for x in X:
                sub_vol_fn = os.path.join(out_folder, out_prefix + '_z' + str(z) + '_y' + str(y) + '_x' + str(x) + '_w' + str(w) + '.tif')
                if os.path.isfile(sub_vol_fn):
#                    existing = tifffile.imread(sub_vol_fn)
#                    if np.array_equal(sub_vol, existing):
#                        print(f'{sub_vol_fn} already exists, skipping')
#                        continue
#                    else:
#                        raise Exception(f'{sub_vol_fn} already exists, but is different')
                    print(f'{sub_vol_fn} already exists, skipping')
                    continue
                sub_vol = data[z:z+w, y:y+w, x:x+w]
                if isinstance(sub_vol, da.Array):
                    sub_vol = sub_vol.compute()
                tifffile.imwrite(sub_vol_fn, sub_vol)

def tiff_to_zarr(tiff_fn, 
                 chunk_size=200, 
                 channel_dim=1, 
                 extract_channel=None, 
                 postfix='', 
                 roi=None, 
                 tif_reader=tifffile.memmap, 
                 verbose=False):
    """Resave a tiff file as zarr archive.
    Ideally reading of the data is done by memmap, which can avoid loading all data into RAM (not tested).
    
    Parameters
    ----------
    tiff_fn: str
        path to tiff file
    chunk_size: int or tuple
        size of chunks in zarr archive
    channel_dim: int or None
        dimension of channel axis
    extract_channel: int or None
        if int, extract the channel at this position
    postfix: str
        name of extracted channel (for naming the zarr archive)
    roi: tuple or None
        bounding box to extract
    tif_reader: function
        function to read the tiff file

    Returns
    -------
    data: dask array
        dask array of the zarr archive

    Notes
    -----
    - The zarr archive is saved in the same directory as the tiff file
    - The zarr archive is named after the tiff file (and, optinally, the extracted channel)
    TODO: adjust postfix to include all operations (e.g. also bbox)
    """
    
    # read
    lazy_imread = delayed(tif_reader)
    reader = lazy_imread(tiff_fn)
    try:
        data = da.from_delayed(reader, 
                            shape=tifffile.memmap(tiff_fn).shape, 
                            dtype=tifffile.memmap(tiff_fn).dtype)
    except:
        data = da.from_array(skio.imread(tiff_fn), chunks=chunk_size)
    # extract channel
    if extract_channel is not None:
        if len(postfix) == 0:
            raise Exception(f'Please specify postfix when extracting a channel')
        ch_roi = [slice(None, None, None) for dim in range(len(data.shape))]
        ch_roi[channel_dim] = slice(extract_channel, 
                                    extract_channel+1, 
                                    None)
        ch_roi = tuple(ch_roi)
        data = np.squeeze(data[ch_roi])
        
    # extract bbox
    if roi is not None:
        data = data[roi]
        
    # chunk
    if isinstance(chunk_size, int):
        chunk_size = [chunk_size] * len(data.shape)
        # if there is a channel dimension and we did not extract a channel
        if channel_dim is not None and extract_channel is None:
            chunk_size[channel_dim] = 1
    data = data.rechunk({i:ch_s for i, ch_s in enumerate(chunk_size)})
    
    # save as zarr
    tiff_extension = '.' + tiff_fn.rsplit('.', 1)[-1]
    zarr_fn = tiff_fn.replace(tiff_extension, f'_{postfix}.zarr')
    if not os.path.isdir(zarr_fn):
        da.to_zarr(data, zarr_fn)
        if verbose:
            print(f'Saved zarr archive to {zarr_fn}')
    else:
        print(f'{zarr_fn} already exists, loading from file')
    
    # re-load from file
    data = da.from_zarr(zarr_fn)
    return data, zarr_fn


   
    

def transpose_tif(tif_path, tif_path_new=None, max_gb=10):
    """Transpose a (large) tif file without loading it into RAM completely"""

    if tif_path_new is None:
        if os.path.isfile(tif_path.replace('.tif', '_T.tif')):
            raise Exception('File already exists.')
        else:
            tif_path_new = tif_path.replace('.tif', '_T.tif')
            print('new filename:', tif_path_new)

    print('original imagej metadata')
    with tifffile.TiffFile(tif_path) as tif:
        print(tif.imagej_metadata)
    print('NOTE: this script does NOT PRESERVE METADATA')

    # Get shape [z, y, x]
    mm_orig = tifffile.memmap(tif_path)
    shape = mm_orig.shape
    # Get dtype
    dtype = mm_orig.dtype

    print('original shape:', str(shape))
    print('original dtype:', str(dtype))

    # TODO: parameterize (enable other dtypes)
    if str(dtype) != '>u2':
        raise Exception('Not sure which dtype to save to')
    else:
        dtype_out = np.uint16
        print('saving as: ', dtype_out)

    # TODO: double-check if this is really necessary
    # With our large .tif-files, tifffile.memmap sometimes returns a 3d array (in this case tif.pages is of length 1),
    #  sometimes a 2d array (first tif page, in this case tif.pages is of length z)
    # Create mms - a list of memmaps for each z-slice of the input data, each of shape [y, x]
    with tifffile.TiffFile(tif_path) as tif:
        # if tifffile can find pages for the file
        if len(tif.pages) == shape[0]:
            # get memmaps for each z-slice of the input data, each of shape [y, x]
            mms = []
            for p in tqdm(range(shape[0])):
                mms.append(tif.pages[p].asarray(out='memmap'))
        # if tifffile can find pages for the file
        else:
            mms = []
            for p in tqdm(range(shape[0])):
                mms.append(mm_orig[p])

    # divide by two because data is in memory twice (once as list, once as array)
    max_gb = max_gb / 2
    max_nbytes = max_gb * 1024 ** 3
    # bytes_per_slice = tifffile.memmap(fn)[..., 0].nbytes
    bytes_per_slice = np.empty(shape[:-1], dtype=dtype).nbytes
    max_slices = int(max_nbytes / bytes_per_slice)

    iterations_needed = int(np.ceil(shape[-1] / max_slices))
    print('need to create', str(shape[-1]), 'slices')
    print('planning to use', str(max_gb * 2), 'GB of RAM')
    print('can process', str(max_slices), 'slices at a time')
    print('read/write iterations needed:', str(iterations_needed))

    # Initialize a new file and memmap for the transposed data
    out_array = np.empty(shape[::-1], dtype=dtype_out)
    # TODO: Can this be initialized more efficiently?
    tifffile.imwrite(tif_path_new, out_array, imagej=True)

    mm_out = tifffile.memmap(tif_path_new)
    # for each (batch of) new though-plane(s)
    start = 0
    for batch in tqdm(range(iterations_needed)):
        # TODO: could initialize im as array
        #  i.e. work without the list and use half the memory
        #  but usually assembling data in lists and then stacking
        #  is much faster than many assignments to a pre-allocated array
        #  so this is probably not worth it
        # TODO: Here we use many read-operations to assemble full pages, which are then written in one write-operation
        #  This is probably efficient because we then write only once to each new tiff page
        #  Could experiment with writing partial pages, which would require fewer read-operations
        # TODO: Create an option to do this with intermediate storage to .zarr
        im = []
        stop = start + max_slices
        if stop > shape[-1] - 1:
            stop = shape[-1] - 1
        for x in tqdm(range(shape[-3])):
            im.append(np.array(mms[x][:, start:stop]))
        # make sure not to flip/mirror the stack
        mm_out[start:stop] = np.stack(im).T.astype(dtype_out)
        mm_out.flush()
        start = stop

    return tif_path_new


def tif_to_zarr(tif_path, zarr_path=None, max_gb=10):
    """Save a tif file as zarr without loading it into RAM completely"""
    if zarr_path is None:
        if os.path.isfile(file.replace('.tif', '_T.tif')):
            raise Exception('File already exists.')
        else:
            zarr_path = file.replace('.tif', '_T.tif')
            print('new filename:', zarr_path)

    print('original imagej metadata')
    with tifffile.TiffFile(tif_path) as tif:
        print(tif.imagej_metadata)
    print('NOTE: this script does NOT PRESERVE METADATA')

    # Get shape [z, y, x]
    mm_orig = tifffile.memmap(tif_path)
    shape = mm_orig.shape
    # Get dtype
    dtype = mm_orig.dtype

    print('original shape:', str(shape))
    print('original dtype:', str(dtype))

    # TODO: parameterize (enable other dtypes)
    if str(dtype) != '>u2':
        raise Exception('Not sure which dtype to save to')
    else:
        dtype_out = np.uint16
        print('saving as: ', dtype_out)

    # TODO: double-check if this is really necessary
    # With our large .tif-files, tifffile.memmap sometimes returns a 3d array (in this case tif.pages is of length 1),
    #  sometimes a 2d array (first tif page, in this case tif.pages is of length z)
    # Create mms - a list of memmaps for each z-slice of the input data, each of shape [y, x]
    with tifffile.TiffFile(tif_path) as tif:
        # if tifffile can find pages for the file
        if len(tif.pages) == shape[0]:
            # get memmaps for each z-slice of the input data, each of shape [y, x]
            mms = []
            for p in tqdm(range(shape[0])):
                mms.append(tif.pages[p].asarray(out='memmap'))
        # if tifffile can find pages for the file
        else:
            mms = []
            for p in tqdm(range(shape[0])):
                mms.append(mm_orig[p])

    # Initialize a new file
    store = zarr.DirectoryStore(zarr_path)
    zarr_f = zarr.create(list(shape)[::-1],
                         chunks=(200, 200, 200),
                         dtype=np.uint16,
                         store=store,
                         overwrite=True)

    # divide by two because data is in memory twice (once as list, once as array)
    max_gb = max_gb / 2
    max_nbytes = max_gb * 1024 ** 3
    # bytes_per_slice = tifffile.memmap(fn)[..., 0].nbytes
    bytes_per_slice = np.empty(shape[:-1], dtype=dtype).nbytes
    max_slices = int(max_nbytes / bytes_per_slice)

    iterations_needed = int(np.ceil(shape[-1] / max_slices))
    print('need to create', str(shape[-1]), 'slices')
    print('planning to use', str(max_gb * 2), 'GB of RAM')
    print('can process', str(max_slices), 'slices at a time')
    print('read/write iterations needed:', str(iterations_needed))

    slices = shape[0]
    for p in tqdm(range(0, range(shape[-1]), max_slices)):
        zarr_f[..., p:np.min([p + 50, max_slices])] = mm[p:np.min([p + max_slices, 500]), ...].T.astype(np.uint16)

    return zarr_path


def get_zarr_size_on_disk(folder):
    """Get the size of a zarr directory on disk (in GB)"""
    n_bytes = sum(os.path.getsize(os.path.join(folder, f))
                  for f in os.listdir(folder)
                  if os.path.isfile(os.path.join(folder, f)))
    return n_bytes / 1024**3

