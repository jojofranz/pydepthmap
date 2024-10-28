# pydepthmap

This repository was created during my time in the [CBClab](https://www.cbclab.org/), for some of the analysis presented in our [angioMASH preprint](https://www.researchgate.netd/publication/383502948_Investigating_microscopic_angioarchitecture_in_the_human_visual_cortex_in_3D_with_angioMASH_tissue_clearing_and_labelling). We used it to profile microscropic structures (cell bodies and blood vessels) in a mesoscopic reference frame (cortical layers). 

## Features
- (Down-/up-)Sampling
    - For (manual) segmentation of large structures (here, the cortex), we downsample our high-res, large-FOV LSFM data, manually segment, create a coordinate systems with [LayNii](https://github.com/layerfMRI/LAYNII), upsample it, and use it as a coordinate system for histogramming our structures of interest (cell bodies and blood vessels)
- (d-dimensional) Histogramming 
    - Using [boost-histogram](https://github.com/scikit-hep/boost-histogram) (or optional numpy or pandas)
- Filtering
    - Equalization and background subtration (adopted from [ClearMAP](https://github.com/ClearAnatomics/ClearMap), now with now optional with sparse structuring elements, clipping, GPU processing)
    [we use this only for visualization]
- (Perhaps most importantly, ) We use [zarr](https://zarr.readthedocs.io/en/stable/), [dask](https://www.dask.org/), and [dask-image](https://image.dask.org/en/latest/) to parallelize many of these steps, and execute them out-of-core (on image-data that is larger than system memory). For this purpose, this repository implements a few wrappers to enable chunked processing (e.g. regionprops (see `core.apply_chunked_regionprops` which computes area and bbox in chuked arra), which also enables connected-component filtering (e.g. see `core.accept_labels`))

## Limitations
This repository was created for a specific project but is slowly(!) being extended to be of more general use. Also: This repository is only a pipeline, useful in the context of many other great tools (in case of our [angioMASH preprint](https://www.researchgate.netd/publication/383502948_Investigating_microscopic_angioarchitecture_in_the_human_visual_cortex_in_3D_with_angioMASH_tissue_clearing_and_labelling): [BigSticher](https://imagej.net/plugins/bigstitcher/), [StarDist](https://github.com/stardist/stardist), [LayNii](https://github.com/layerfMRI/LAYNII), [localthickness](https://pypi.org/project/localthickness/), [ITKSnap](http://www.itksnap.org/pmwiki/pmwiki.php), [Napari](https://napari.org/stable/), [ParaView](https://www.paraview.org/))

## Use
See the notebook `angioMASH_preprint.ipynb` for an example. We use Docker. This has the advantage that we only need to install minimal requirements into our OS (we use Ubuntu) - which is especially helpful when using nvidia-gpus. For using the GPU, please make sure to first install the [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit).

Then, get a suitable docker image. The image used below was selected to use a rather old nvidia-gpu - depending on your hardware, consider using a more recent image. 
```
docker pull tensorflow/tensorflow:2.10.1-gpu-jupyter
```

Then, we start a notebook inside the container (in interactive mode, using gpu, with a mounted volume that should hold our data):
```
docker run -it --runtime=nvidia --gpus all   -v /media/johannes/Compute2TB4/angio:/tf/angio -p 8888:8888 tensorflow/tensorflow:2.10.1-gpu-jupyter
```

Inside the notebook, install a few requirements. First for using opencv:
```
! apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
```

Then, all the rest:
```
! pip install -r pydepthmap/requirements.txt
```
