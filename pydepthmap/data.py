import os
import numpy as np
import zarr
import dask.array as da
import tifffile
from skimage.io import imread
import nibabel as nb
import pandas as pd
import copy
import subprocess
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
from .utils import (tiff_to_zarr, 
                    mip_stride, 
                    to_nii, 
                    from_file, 
                    get_LayNii)
from .core import (proc_cortex_seg, 
                   seg_to_rim, 
                   proc_LayNii, 
                   proc_metric_equidist, 
                   dask_zoom, 
                   apply_chunked_regionprops, 
                   accept_labels, 
                   measure_hist, 
                   measure_density, 
                   postprocess_density, 
                   volume_from_cellcenters, 
                   get_manual_layers)
from dask_image.ndmeasure import label as da_label
from local_thickness_dahl.localthickness import local_thickness as apply_local_thickness_dahl

#TODO: move to utils
from edt import edt

class DataSet():
    def __init__(self, general, raw_entries, verbose=False):
        """Placeholder for diverse information about a dataset.

        A dataset is a collection of entries, which can be raw data or derivatives.
        This class attempts to hold as little data as possible: 
        When possible, it uses memmap or dask/zarr-arrays on disk, but it can also hold numpy-arrays in memory.

        This class:
        1)  holds rudimentary meta-data together with a rudimentary graph structure for data provenance
            NOTE: There is some flexibility wrt. which meta-data is mandatory (see general['description'])
        2)  provides methods to add entries and derivatives
        3)  provides methods to access data
        4)  provides methods to process data and create derivatives
        
        Initialization is done with general parameters and a list of 
        (only raw-data!) entries. Derivatives can be added later.

        Generally, parameters can be set through the dictionaries: general and entries.
        Specific parameters in entries overwrite general parameters.
        Some parameters need to be GENERAL.
        Some parameters need to be SPECIFIC to entries.
        Some parameters can be general OR specific to entries.
        Among the specific parameters, 'resave_mod' is used to re-save .tif files as zarr-archives 
        and, optionally, restrict this to regions-of-interests.
        NOTE: When resaving, it is preferred to load previous results from zarr-archives. 
              When an archive with a fitting name is found, we load it - and do NOT check if it is up-to-date.
        Among the specific parameters, 'load_mod' is used to load data 
        and, optionally, restrict this to regions-of-interests.

        NOTE: Alwary access data through get_data, as this accounts for load_mod.
        Note: Always access data in files through get_data_from_files, as this accounts for load_mod.

        Parameters
        ----------
        general : dict
            folder : str
                MANDATORY, path to working directory
                all file-paths are relative to this folder
            description : list of str
                MANDATORY, exhaustive description of the raw dataset
                e.g. ['region', 'channel'] for a dataset where all raw data can 
                be uniquely identified by the flags: region and channel
                derivatives get the additional flag: special
                description['special'] can then refer to anything
            NOTE: additional parameters can be added
        entries : list of dict
                one list-entry for each raw data
                    one entry for each flag in general['description'] : str
                        MANDATORY, for each flag in general['description']
                        e.g. 'region' or 'channel'
                    special : str
                        PROHIBITED - this is a reserved keyword for derivatives (prohibited for the raw-data-entries that are used to intitialize a dataset)
                    files : dict
                        optional, path(s) to file(s) (relative to general.folder)
                        if not provided, data needs to be provided in entry.data
                    resave_mod : dict
                        optional, parameters to resave the data
                        resave_mod['roi'] : list of nD 2-element (start, stop) lists or None
                            optional, region of interest to extract
                            if None, do not extract a region of interest
                            e.g. [[100, 200], [100, 200], [100, 200]]
                        resave_mod['chunk_size'] : int or None
                            optional, size of chunks in zarr archive
                            if None, use default chunk size
                    load_mod : dict
                        optional, parameters to load the data
                        load_mod['roi'] : list of nD 2-element (start, stop) lists or None
                            optional, region of interest to extract
                            if None, do not extract a region of interest
                            e.g. [[100, 200], [100, 200], [100, 200]]
                        load_mod['chunk_size'] : int or None
                            optional, size of chunks in dask array
                            if None, use default chunk size
                            only used if data is a da.Array
                    data : dask.array or numpy.array or tifffile.memmap
                        optional, data array
                        if not provided, data needs to be provided in entry['files']
                    measurements : dict
                        optional, measurements of the data
                    parent : str or None
                        PROHIBITED - this is a reserved keyword for derivatives (prohibited for the raw-data-entries that are used to intitialize a dataset)
                    children : list of str
                        PROHIBITED - this is a reserved keyword for derivatives (prohibited for the raw-data-entries that are used to intitialize a dataset)

        Returns
        -------
        DataSet object

        Notes
        -----

        """

        reserved_keywords = ['folder',  # mandatory, general
                             'description',  # mandatory, general
                             'special',  # optional, entry
                             'files',  # optional, entry
                             'resave_mod',  # optional, entry
                             'load_mod',  # optional, entry
                             'data',  # optional, entry
                             'measurements',  # optional, entry
                             'parent',  # optional, entry
                             'children']  # optional, entry
        resave_mod_keywords = ['roi', 
                               'chunk_size']
        load_mod_keywords = ['roi',
                             'chunk_size']
        for key in general['description']:
            if key in reserved_keywords:
                raise Exception(f'Key: {key} is reserved as a general keyword and cannot be used as part of the description')
        self.reserved_keywords = reserved_keywords
        self.resave_mod_keywords = resave_mod_keywords
        self.load_mod_keywords = load_mod_keywords

        self.verbose = verbose

        if not 'folder' in general.keys():
            raise Exception('Need to provide folder in general (used as working directory)')
        if not 'description' in general.keys():
            raise Exception('Need to provide description in general (exhaustive description of the raw dataset)')

        self.general = general
        
        folder = general['folder']
        if not os.path.exists(folder):
            raise Exception(f'Folder {folder} does not exist')
        self.folder = folder
        
        self.entries = {}
        for entry in raw_entries:
            if 'special' in entry.keys():
                raise Exception('special is a reserved keyword for derivatives, cannot add it to raw-data entries')
            if 'parent' in entry.keys():
                raise Exception('parent is a reserved keyword for derivatives, cannot add it to raw-data entries')
            if 'children' in entry.keys():
                raise Exception('children is a reserved keyword, it is added automatically when addid an entry with a parent')
            name = self._add_entry(entry)
            
    def __str__(self):
        return ' '.join(list(self.entries.keys()))
    
    def show_zarr(self):
        for name, entry in self.entries.items():
            if 'files' in entry.keys():
                if 'zarr' in entry['files'].keys():
                    print(f'entry: {name}')
                    display(self.get_data(name))  # Use get_data to account for load_mod
    
    def add_derivative(self, entry, parent=None):
        """Add a derivative to the dataset.

        Parameters
        ----------
        entry : dict
            description of the entry
        parent : str or None
            name of the parent entry
            if None, parent needs to be provided through the entry-dict
        """
        # get parent if it was provided through the entry-dict
        # BUT preferentially choose the description given throught he seperate partent-argument
        if parent is None:
            if 'parent' in entry.keys():
                parent = copy.deepcopy(entry['parent'])
                entry.pop('parent')
        else:
            if 'parent' in entry.keys():
                entry.pop('parent')

        if parent is None:
            raise Exception('Need to provide parent')
        if parent not in self.entries.keys():
            raise Exception('Parent not found in entries')
                
        name = self._add_entry(entry, parent=parent)
        self.entries[parent]['children'].append(name)

        return name

        
    def _add_entry(self, entry, parent=None):
        """Add an entry to the dataset.

        Parameters
        ----------
        entry : dict
            description of the entry
        parent : str or None
            name of the parent entry

        Note:
        -----
        - For internal use: Use add_derivative instead
        """
        # iteratively copy from entry to new_entry, while checking 
        # for mandatory parameters
        new_entry = {}

        if parent is None:
            # raw-data
            if 'special' in entry.keys():
                raise Exception('special is a reserved keyword for derivatives, cannot add it to raw-data entries')
#            # TODO: soften requirement for raw-data, e.g. allow other sources than raw.tif
#            if not 'raw.tif' in entry['files'].keys():
#                raise Exception('Need to provide raw.tif for each entry in entries')

        if not ('files' in entry.keys() or 'data' in entry.keys()):
            raise Exception('Need to provide at least one out of files or data for each entry in entries')

        # check for description
        for key in self.general['description']:
            if key in self.reserved_keywords or key in self.resave_mod_keywords or key in self.load_mod_keywords:
                raise Exception(f'Key: {key} is reserved as a general keyword and cannot be used as part of the description')
            if not key in entry.keys():
                raise Exception(f'Need to provide {key} for each entry in entries')
            else:
                new_entry[key] = entry[key]
                entry.pop(key)

        # TODO: add validity check for each key 

        # over the parameters that are not part of the description
        for key in self.reserved_keywords:
            # load from entry
            if key in entry.keys():
                new_entry[key] = entry[key]
                entry.pop(key)
            # if not in entry, load from general
            elif key in self.general.keys():
                new_entry[key] = copy.deepcopy(self.general[key])
            # if not provided AND if optional
            # intitialize with None
            elif key in ['special', 
                         'files',
                         'resave_mod', 
                         'load_mod',
                         'data', 
                         'parent']:
                if not key in entry.keys():
                    new_entry[key] = None
            # if not provided AND if optional
            # initialize with []
            elif key in ['children']:
                new_entry[key] = []
            # intitialize with {}  
            elif key in ['measurements']:
                new_entry[key] = {}
            else:
                raise Exception(f'Need to provide {key} in general or entries')

        # if there are still keys left in entry, raise exception
        if len(entry.keys()):
            raise Exception(f'Unknown key(s) in entry: {entry}')

        # check for resave_mod parameters
        if new_entry['resave_mod'] is not None:
            resave_mod = copy.deepcopy(new_entry['resave_mod'])
            for key in self.resave_mod_keywords:
                # if not provided, use default
                if not key in new_entry['resave_mod'].keys():
                    if key == 'roi':
                        new_entry['resave_mod'][key] = None
                    elif key == 'chunk_size':
                        new_entry['resave_mod'][key] = None
                # if provided, check if it is allowed
                else:
                    resave_mod.pop(key)

            # if there are still keys left in resave_mod, raise exception
            if len(resave_mod.keys()):
                raise Exception(f'Unknown key(s) in resave_mod: {resave_mod}')
            
        # check for load_mod parameters
        if new_entry['load_mod'] is not None:
            load_mod = copy.deepcopy(new_entry['load_mod'])
            for key in self.load_mod_keywords:
                # if not provided, use default
                if not key in new_entry['load_mod'].keys():
                    if key == 'roi':
                        new_entry['load_mod'][key] = None
                    elif key == 'chunk_size':
                        new_entry['load_mod'][key] = None
                # if provided, check if it is allowed
                else:
                    load_mod.pop(key)

            # if there are still keys left in load_mod, raise exception
            if len(load_mod.keys()):
                raise Exception(f'Unknown key(s) in load_mod: {load_mod}')
                
        # create (unique) name
        name = '_'.join([f'{new_entry[key]}' for key in new_entry['description']])
        if parent is not None:
            if 'special' in new_entry.keys():
                name = f'{name}_{new_entry["special"]}'
            else:
                raise Exception('Need to provide special when adding a derivative')
        new_entry['name'] = name
        if name in self.entries.keys():
            raise Exception(f'Entry {name} already exists')

        if new_entry['resave_mod'] is not None:
            # convert roi to tuple of slices
            # TODO: allow step
            # TODO make explicit that when resave_mod or load_mod is given, both roi and chunk_size are required together
            roi = new_entry['resave_mod']['roi']
            if isinstance(roi, list):
                new_entry['resave_mod']['roi'] = tuple([slice(slc[0], slc[1], None) for slc in roi])

            if new_entry['resave_mod']['chunk_size'] is None:
                new_entry['resave_mod']['chunk_size'] = 200
            if isinstance(new_entry['resave_mod']['chunk_size'], int):
                new_entry['resave_mod']['chunk_size'] = [new_entry['resave_mod']['chunk_size']] * len(new_entry['resave_mod']['roi'])

        if new_entry['load_mod'] is not None:
            # convert roi to tuple of slices
            # TODO: allow step
            roi = new_entry['load_mod']['roi']
            if isinstance(roi, list):
                new_entry['load_mod']['roi'] = tuple([slice(slc[0], slc[1], None) for slc in roi])

            if new_entry['load_mod']['chunk_size'] is None:
                new_entry['load_mod']['chunk_size'] = 200
            if isinstance(new_entry['load_mod']['chunk_size'], int):
                new_entry['load_mod']['chunk_size'] = [new_entry['load_mod']['chunk_size']] * len(new_entry['load_mod']['roi'])
            
        # add parent
        if parent is not None:
            new_entry['parent'] = parent
        # NOTE: this entry gets added as a child to the parent through add_derivative

        # convert raw tif to zarr
        # TODO: parameterize if conversion is done
        if new_entry['resave_mod'] is not None:
            if 'tif' in new_entry['files'].keys():
                new_entry = self.convert_tif_to_zarr(name, new_entry)
            else:
                raise Exception('Need to provide a tif file when providing resave_mod')

        # add entry
        self.entries[name] = new_entry            
        if self.verbose:
            print(f'Added entry: {name}\n')
        return name 
    
    def load_zarr(self, name, entry):
        """
        """
        if self.verbose:
            print(f'Loading {name} from zarr')
        data = da.from_zarr(entry['files']['zarr'])
        if entry['load_mod'] is not None:
                if entry['load_mod']['roi'] is not None:
                    roi = entry['load_mod']['roi']
                    data = data[roi].squeeze()
        return entry
    
    def convert_tif_to_zarr(self, name, entry):
        """
        """
        if self.verbose:
            print(f'Converting {name} to zarr')
        tiff_fn = os.path.join(self.folder, entry['files']['tif'])
        data, zarr_fn = tiff_to_zarr(tiff_fn, 
                                     chunk_size=entry['resave_mod']['chunk_size'], 
                                     postfix=entry['channel'], 
                                     roi=entry['resave_mod']['roi'], 
                                     verbose=self.verbose)
        entry['files']['zarr'] = zarr_fn
        return entry
    
    def get_data(self, name):
        """
        """
        if name not in self.entries.keys():
            raise Exception(f'Entry {name} not found')
        if self.entries[name]['data'] is None:
            return self.get_data_from_files(name)
        else:
            data = self.entries[name]['data']
            if self.entries[name]['load_mod'] is not None:
                if self.entries[name]['load_mod']['roi'] is not None:
                    roi = self.entries[name]['load_mod']['roi']
                    data = data[roi].squeeze()
            return data
    
    def get_data_from_files(self, name):
        """Load data from files.
        Tries to load data from zarr, then from tif, then from nii.
        If possible return dask.array, else np.memmap, else numpy.array.

        Parameters
        ----------
        name : str
            name of entry

        Returns
        -------
        data : dask.array or numpy.array
            data from files

        Notes
        -----
        Prefer get_data over get_data_from_files.
        """
        if name not in self.entries.keys():
            raise Exception(f'Entry {name} not found')
        if 'zarr' in self.entries[name]['files'].keys():
            data = da.from_zarr(self.entries[name]['files']['zarr'])
#        if 'raw.tif' in self.entries[name]['files'].keys():
#            try:
#                return tifffile.memmap(self.entries[name]['files']['raw.tif'])
#            except:
#                return imread(self.entries[name]['files']['raw.tif'])
        elif 'tif' in self.entries[name]['files'].keys():
            try:
                data = tifffile.memmap(self.entries[name]['files']['tif'])
            except:
                data = imread(self.entries[name]['files']['tif'])
        elif 'nii' in self.entries[name]['files'].keys():
            data = nb.load(self.entries[name]['files']['nii']).get_fdata()
        else:
            if len(self.entries[name]['files']):
                raise Exception(f'Could not load {name} from {self.entries[name]["files"]}')
            else:
                raise Exception(f'No file for entry {name}')
            
        if self.entries[name]['load_mod'] is not None:
            if self.entries[name]['load_mod']['roi'] is not None:
                roi = self.entries[name]['load_mod']['roi']
                data = data[roi].squeeze()
            if isinstance(data, da.Array):
                if self.entries[name]['load_mod']['chunk_size'] is not None:
                    chunk_size = self.entries[name]['load_mod']['chunk_size']
                    if tuple(chunk_size) != data.chunksize:
                        data = data.rechunk(chunk_size)
        return data


    def show_graph(self):
        """ Visualize the dataset-entries as a graph.
        """

        G = nx.Graph()

        nodes = []#list(ds.entries.keys())
        subgraph = []
        origins = []
        is_origin = []
        edges = []
        for entry in self.entries.keys():
            nodes.append(entry)
            subgraph.append(self.get_grandparent(entry, return_self=True))
            parent = self.entries[entry]['parent']
            if parent is not None:
                edges.append([parent, entry])
            else:
                origins.append(entry)
            is_origin.append(parent is None)
            children = self.entries[entry]['children']
            for child in children:
                edges.append([entry, child])
        edges = np.array(edges)
        edges = pd.DataFrame({'from': edges[:, 0], 
                            'to': edges[:, 1]})
        edges = edges.values.tolist()

        G.add_nodes_from(nodes)
        nx.set_node_attributes(G, {i: v for i, v in zip(nodes, subgraph)}, 'subgraph')
        nx.set_node_attributes(G, {i: v for i, v in zip(nodes, is_origin)}, 'is_origin')
        color = ['red' if i_o else 'lightblue' for i_o in is_origin]
        nx.set_node_attributes(G, {i: v for i, v in zip(nodes, color)}, 'color')
        G.add_edges_from(edges)

        for entry_k, entry_v in self.entries.items():
            subgraph = self.get_grandparent(entry_k, return_self=True)
            if not entry_v['measurements'] == {}:
                for measurement in entry_v['measurements']:
                    node_id = f'{entry_k}_{measurement}'
                    G.add_node(node_id, **{'subgraph': subgraph, 
                                        'is_origin': False, 
                                        'color': 'yellow'})
                    G.add_edge(entry_k, node_id)
                    
        
        net = Network(notebook = True, cdn_resources = "remote",
                        bgcolor = "#222222",
                        font_color = "white",
                        height = "750px",
                        width = "100%",
                    filter_menu = True
        )

        nx.set_node_attributes(G, {i: 25 for i in G.nodes}, 'size')

        net.from_nx(G)
        net.show_buttons()
        return net.show('graph.html')
    

    def get_basename(self, name):
        """
        """
        if name not in self.entries.keys():
            raise Exception(f'Entry {name} not found')
        if 'parent' not in self.entries[name].keys():
            return name
        elif self.entries[name]['parent'] is None:
            return name
        else:
            return self.get_grandparent(name)

    def get_base_filename(self, name):
        """
        """
        if name not in self.entries.keys():
            raise Exception(f'Entry {name} not found')
#        if 'raw.tif' in self.entries[name]['files'].keys():
#            fn = self.entries[name]['files']['raw.tif']
#            if fn.lower().endswith('.tif'):
##                return fn[:-len('.tif')]
 #           elif fn.lower().endswith('.tiff'):
 #               return fn[:-len('.tiff')]
 #           else:
 #               raise Exception(f'Unknown tif file extension for {fn}')
        if 'tif' in self.entries[name]['files'].keys():
            fn = self.entries[name]['files']['tif']
            if fn.lower().endswith('.tif'):
                return fn[:-len('.tif')]
            elif fn.lower().endswith('.tiff'):
                return fn[:-len('.tiff')]
            else:
                raise Exception(f'Unknown tif file extension for {fn}')
        elif 'zarr' in self.entries[name]['files'].keys():
            fn = self.entries[name]['files']['zarr']
            if fn.lower().endswith('.zarr'):
                return fn[:-len('.zarr')]
            elif os.path.sep in fn:
                # TODO: hacky
                return fn.split(os.path.sep)[0]
            else:
                raise Exception(f'Unknown zarr file extension for {fn}')
        elif 'nii' in self.entries[name]['files'].keys():
            fn = self.entries[name]['files']['nii']
            if fn.lower().endswith('.nii'):
                return fn[:-len('.nii')]
            elif fn.lower().endswith('.nii.gz'):
                return fn[:-len('.nii.gz')]
            else:
                raise Exception(f'Unknown nii file extension for {fn}')
        else:
            raise Exception(f'No file for entry {name}')    

    def get_measurements(self, name):
        """
        """
        entry = self.get_entry(name)
        if 'measurements' not in entry.keys():
            raise Exception(f'No measurements for entry {name}')
        return entry['measurements']

    def get_entry(self, name):
        """
        """
        if name not in self.entries.keys():
            raise Exception(f'Entry {name} not found')
        return self.entries[name]
    
    def get_parent(self, name):
        """
        """
        if name not in self.entries.keys():
            raise Exception(f'Entry {name} not found')
        if 'parent' not in self.entries[name].keys():
            raise Exception(f'Entry {name} has no parent')
        return self.entries[self.entries[name]['parent']]
    
    def get_children(self, name):
        """
        """
        if name not in self.entries.keys():
            raise Exception(f'Entry {name} not found')
        if 'children' not in self.entries[name].keys():
            raise Exception(f'Entry {name} has no children')
        return self.entries[self.entries[name]['children']]
    
    def get_grandparent(self, name, return_self=False):
        """
        """
        if name not in self.entries.keys():
            raise Exception(f'Entry {name} not found')
        if 'parent' in self.entries[name].keys():
            if self.entries[name]['parent'] is not None:
                parent = self.entries[name]['parent']
                has_parent = True
            else:
                if return_self:
                    return name
                else:
                    raise Exception(f'Entry {name} has no parent')
        else:
            raise Exception(f'Entry {name} has no parent')
        while has_parent:
            if 'parent' in self.entries[parent].keys():
                if self.entries[parent]['parent'] is not None:
                    parent = self.entries[parent]['parent']
                    continue
            has_parent = False
        return parent
    
    def inherit(self, parent, entry):
        """Inherit parameters from parent entry.
        The parameters that are inherited is quite limited.
        currently only the description.
        """
        if parent not in self.entries.keys():
            raise Exception(f'Entry {parent} not found')
        for key in self.general['description']:
            if key not in entry.keys():
                entry[key] = copy.deepcopy(self.entries[parent][key])
        return entry
        
    def add_mip_stride(self, name, mip_r, mip_step, stride, postfix, special='mip_stride', postprocess=None, overwrite=False):
        """
        """
        basename = self.get_basename(name)
        new_name = f'{basename}_{special}'
        if new_name in self.entries.keys():
            raise Exception(f'Entry {new_name} already exists')
        
        if not postfix.endswith('.nii'):
            postfix += '.nii'
        out_fn = self.get_base_filename(name) + postfix
        if os.path.exists(out_fn) and not overwrite:
            fn = out_fn
        else:
            im = mip_stride(self.get_data(name), mip_r, mip_step, stride, postprocess=None)
            fn = to_nii(im, self.get_base_filename(name), postfix, overwrite=overwrite)

        entry = {'files': {'nii': fn}, 
                 'special': special, 
                 'parent': name}
        entry = self.inherit(name, entry)
        new_name = self.add_derivative(entry)

        return new_name
    
    def add_rim_and_metric(self, name, downsampling, dilation, smoothing, 
                   special='metric.equidist.valid', overwrite=False, show=False):
        """Process cortex segmentation and convert to rim.
        This involves a few steps:
            - process cortex segmentation
                - downsampling (to isotropic voxel size)
                - dilation of missing-data label
                - smoothing
            - convert to rim
                - save as nii
                - add to entries
            - cortical depth estimation w. LayNii
                - save as nii
                - add to entries

        Parameters
        ----------
        name : str
            name of entry
        downsampling : list of 3 int (for 3D data)
            downsampling factors
        dilation : int
            The diameter of the dilation ball
        smoothing : int
            The sigma of the gaussian smoothing
        special : str
            special identifier for this derivative
        overwrite : bool
            whether to overwrite existing file

        Returns
        -------
        None

        """
        if name not in self.entries.keys():
            raise Exception(f'Entry {name} not found')

        # TODO: make the reservation of rim as special more explicit
        if special == 'rim':
            raise Exception(f'{special} is a reserved special-keyword for derivatives, cannot use it as special')

        basename = self.get_basename(name)
        new_name = f'{basename}_{special}'
        if new_name in self.entries.keys():
            raise Exception(f'Entry {new_name} already exists')
        
        # TODO: use these as input to to_nii, rather than obscurely re-creating them inside to_nii
        rim_fn = self.get_base_filename(name) + '_rim.nii'
        out_fn = self.get_base_filename(name) + '_' + special + '.nii'

        data = self.get_data(name)
        if os.path.exists(rim_fn) and os.path.exists(out_fn) and not overwrite:
            if self.verbose:
                print(f'Found previous files: ')
                print(f'\t rim - {rim_fn}')
                print(f'\t metric_equidist_valid - {out_fn}')
            rim = from_file(rim_fn)
            metric_equidist_valid = from_file(out_fn)
        else:
            if self.verbose:
                print(f'Could not find previous files: ')
                print(f'\t rim - {rim_fn}')
                print(f'\t metric_equidist_valid - {out_fn}')
                print(f'Processing {name} ...')
            proc = proc_cortex_seg(data, downsampling, dilation=5, smoothing=5)
            rim = seg_to_rim(proc)
            rim_fn = to_nii(rim, self.get_base_filename(name), '_rim', overwrite=overwrite)
            if self.verbose:
                print(f'Created rim: {rim_fn}')
            metric_equidist = proc_LayNii(rim_fn, verbose=self.verbose)
            metric_equidist_valid = proc_metric_equidist(metric_equidist, proc, rim)
            out_fn = to_nii(metric_equidist_valid, self.get_base_filename(name), '_' + special, overwrite=overwrite)
            if self.verbose:
                print(f'Created metric_equidist_valid: {out_fn}')

        entry = {'files': {'nii': rim_fn}, 
                 'special': 'rim', 
                 'parent': name}
        entry = self.inherit(name, entry)
        self.add_derivative(entry)

        entry = {'files': {'nii': out_fn}, 
                 'special': special, 
                 'parent': name}
        entry = self.inherit(name, entry)
        new_name = self.add_derivative(entry)

        if show:
            # TODO: parameteri the roi
            slc = 10
            roi = [slice(None, None, None)] * 3
            roi[-1] = slice(slc, slc+1, None)
            roi = tuple(roi)

            fig, ax = plt.subplots(3, 1, figsize=(10, 10))
            fig.suptitle(f'{name}')
            ax[0].imshow(data[:, ::5, ::5][roi].squeeze())
            ax[0].set_title('raw segmentation')
            ax[1].imshow(rim[roi].squeeze())
            ax[1].set_title('rim')
            ax[2].imshow(metric_equidist_valid[roi].squeeze())
            ax[2].set_title('metric equidistant valid')
            plt.show()

        return new_name
    
    def add_zoom(self, name, real_zoom, output_chunks=200, special='metric.upsampled', try_read=True, overwritecompute=False):
        """Zoom data and add as derivative.

        Parameters
        ----------
        name : str
            name of entry
        real_zoom : list of 3 floats
            zoom factors
        output_chunks : int or list of 3 int
            chunk size of output
        special : str
            special identifier for this derivative
        try_read : bool
            whether to try to read from file
        overwritecompute : bool
            whether to overwrite existing file
            only relevant if try_read is False

        Returns
        -------
        None

        Notes
        -----
        - This function is a wrapper around dask_zoom
        """
        if name not in self.entries.keys():
            raise Exception(f'Entry {name} not found')
        
        basename = self.get_basename(name)
        new_name = f'{basename}_{special}'
        if new_name in self.entries.keys():
            raise Exception(f'Entry {new_name} already exists')
        
        has_file = False
        zoomed_fn = self.get_base_filename(name) + '_' + special + '.zarr'
        if os.path.exists(zoomed_fn) and try_read:
            if self.verbose:
                print(f'Found previous file: {zoomed_fn}')
            if output_chunks is not None:
                print(f'Found previous file: {zoomed_fn}, NOT USING output_chunks {output_chunks}')
            zoomed = da.from_zarr(zoomed_fn)
            has_file = True
        else:
            if self.verbose:
                print(f'Could not find previous file: {zoomed_fn}, computing ...')
            data = self.get_data(name)
            zoomed = dask_zoom(data, real_zoom, output_chunks, order=1)

            if overwritecompute:
                da.to_zarr(zoomed, zoomed_fn, overwrite=True)
                has_file = True

        if has_file:
            entry = {'files': {'zarr': zoomed_fn}, 
                     'special': special, 
                     'parent': name}
        else:
            entry = {'special': special, 
                     'parent': name}
        entry = self.inherit(name, entry)
        new_name = self.add_derivative(entry)

        return new_name
    

    def add_load_mod(self, name, load_mod):
        """
        NOTE: this roi is applied to the data only, not to the file
        """

        if name not in self.entries.keys():
            raise Exception(f'Entry {name} not found')
        
        if self.entries[name]['load_mod'] is not None:
            raise Exception(f'load_mod already exists for entry {name}')
        if 'roi' not in load_mod.keys():
            raise Exception(f'Need to provide roi in load_mod')
        else:
            roi = load_mod['roi']
            if isinstance(roi, list):
                load_mod['roi'] = tuple([slice(slc[0], slc[1], None) for slc in roi])
        if 'chunk_size' not in load_mod.keys():
            load_mod['chunk_size'] = 200
        if isinstance(load_mod['chunk_size'], int):
            load_mod['chunk_size'] = [load_mod['chunk_size']] * 3  # TODO: generalize to nD

        self.entries[name]['load_mod'] = load_mod
        data = self.get_data(name)
        data = data[load_mod['roi']].squeeze()
        if isinstance(data, da.Array):
            if data.chunksize != tuple(load_mod['chunk_size']):
                data = data.rechunk(load_mod['chunk_size'])
        return None
    
    
    def add_strided_downsampling(self, name, data, downsampling, special='strided.downsampled', try_read=True, overwritecompute=False):
        """
        """
        if name not in self.entries.keys():
            raise Exception(f'Entry {name} not found')
        
        basename = self.get_basename(name)
        new_name = f'{basename}_{special}'
        if new_name in self.entries.keys():
            raise Exception(f'Entry {new_name} already exists')
        
        has_file = False
        downsampled_fn = self.get_base_filename(name) + '_' + special + '.zarr'
        if os.path.exists(downsampled_fn) and try_read:
            if self.verbose:
                print(f'Found previous file: {downsampled_fn}')
            downsampled = da.from_zarr(downsampled_fn)
            has_file = True
        else:
            if self.verbose:
                print(f'Could not find previous file: {downsampled_fn}, computing ...')
            
            downsampling = np.array(downsampling)
            z = np.array(data.shape)
            # assure symmetric placement of windows (up to integer rounding)
            start = np.floor((z/2) % downsampling).astype(int)
            roi = tuple([slice(strt, None, step) for strt, step in zip(start, downsampling)])
            downsampled = data[roi]
            # avoid irregular chunking (problem when saving to zarr)
            downsampled = downsampled.rechunk(downsampled.chunksize)

            if overwritecompute:
                da.to_zarr(downsampled, downsampled_fn, overwrite=True)
                has_file = True

        if has_file:
            entry = {'files': {'zarr': downsampled_fn}, 
                     'special': special, 
                     'parent': name}
        else:
            entry = {'special': special, 
                     'parent': name}
        entry = self.inherit(name, entry)
        new_name = self.add_derivative(entry)

        return new_name


    def add_label(self, name, special='label', overwrite=False):
        """
        """
        basename = self.get_basename(name)
        new_name = f'{basename}_{special}'
        if new_name in self.entries.keys():
            raise Exception(f'Entry {new_name} already exists')
        
        label_fn = self.get_base_filename(name) + '_' + special + '.zarr'
        if os.path.exists(label_fn) and not overwrite:
            if self.verbose:
                print(f'Found previous file: {label_fn}')
            label = da.from_zarr(label_fn)
        else:
            if self.verbose:
                print(f'Could not find previous file: {label_fn}, computing ...')
            data = self.get_data(name)
            label, n_labels = da_label(data)
            da.to_zarr(label, label_fn, overwrite=True)

        entry = {'files': {'zarr': label_fn}, 
                 'special': special, 
                 'parent': name}
        entry = self.inherit(name, entry)
        new_name = self.add_derivative(entry)
        
        return new_name
    
    
    def measure_regionprops(self, name, postfix='regionprops'):
        """
        """
        if name not in self.entries.keys():
            raise Exception(f'Entry {name} not found')
        out_fn = '_'.join([name, postfix + '.csv'])
        if os.path.exists(out_fn):
            if self.verbose:
                print(f'Found previous file: {out_fn}')
            regionprops = pd.read_csv(out_fn)
        else:
            if self.verbose:
                print(f'Could not find previous file: {out_fn}, computing ...')
            data = self.get_data(name)
            regionprops = apply_chunked_regionprops(data)
            regionprops.to_csv(out_fn)
        self.entries[name]['measurements'][postfix] = out_fn
        if self.verbose:
            print(f'Added measurements: {postfix} to entry {name}\n')
        return regionprops
    

    def add_CCfilter(self, name, min_area=None, max_area=None, special='CCfilter'):
        """Connected-component area filter.
        Note: assumes that regionprops have been computed and the filename is 
        stored in entries[name]['measurements']['regionprops']
        """
        basename = self.get_basename(name)
        new_name = f'{basename}_{special}'
        if new_name in self.entries.keys():
            raise Exception(f'Entry {new_name} already exists')
        
        filtered_fn = self.get_base_filename(name) + '_' + special + '.zarr'
        if os.path.exists(filtered_fn):
            if self.verbose:
                print(f'Found previous file: {filtered_fn}')
            filtered = da.from_zarr(filtered_fn)
        else:
            if self.verbose:
                print(f'Could not find previous file: {filtered_fn}, computing ...')

            if 'regionprops' in self.entries[name]['measurements'].keys():
                regionprops_fn = self.entries[name]['measurements']['regionprops']
                regionprops = pd.read_csv(regionprops_fn)

                if min_area is None and max_area is None:
                    raise Exception('Need to provide at least one, min_area or max_area')
                if min_area is None:
                    min_area = regionprops['area'].min()
                if max_area is None:
                    max_area = regionprops['area'].max()
                include = regionprops.loc[np.logical_and(regionprops['area'] >= min_area, 
                                                         regionprops['area'] <= max_area), 'label']
                include = np.array(include)
                
                data = self.get_data(name)

                dtype = data.dtype
                filtered = data.map_blocks(accept_labels, 
                                           dtype=dtype, 
                                           include=include)

                da.to_zarr(filtered, filtered_fn, overwrite=True)
            else:
                raise Exception('Need to compute regionprops before filtering connected components')

        entry = {'files': {'zarr': filtered_fn}, 
                 'special': special, 
                 'parent': name}
        entry = self.inherit(name, entry)
        new_name = self.add_derivative(entry)

        return new_name
    

    def add_combined_segmentation(self, name1, name2, special='combined.segmentation'):
        """
        """
        basename = self.get_basename(name1)
        new_name = f'{basename}_{special}'
        if new_name in self.entries.keys():
            raise Exception(f'Entry {new_name} already exists')
        
        combined_fn = self.get_base_filename(name1) + '_' + special + '.zarr'
        if os.path.exists(combined_fn):
            if self.verbose:
                print(f'Found previous file: {combined_fn}')
            combined = da.from_zarr(combined_fn)
        else:
            if self.verbose:
                print(f'Could not find previous file: {combined_fn}, computing ...')
            data1 = self.get_data(name1)
            data2 = self.get_data(name2)
            combined = np.logical_or(data1>0, data2>0)
            da.to_zarr(combined, combined_fn, overwrite=True)

        entry = {'files': {'zarr': combined_fn}, 
                 'special': special, 
                 'parent': name1}
        entry = self.inherit(name1, entry)
        new_name = self.add_derivative(entry)

        return new_name


    def add_lt_transform(self, name, depth=(50,50, 50), scale=.5, special='LTtransform'):
        """
        """
        basename = self.get_basename(name)
        new_name = f'{basename}_{special}'
        if new_name in self.entries.keys():
            raise Exception(f'Entry {new_name} already exists')
        
        lt_transform_fn = self.get_base_filename(name) + '_' + special + '.zarr'
        if os.path.exists(lt_transform_fn):
            if self.verbose:
                print(f'Found previous file: {lt_transform_fn}')
            lt = da.from_zarr(lt_transform_fn)
        else:
            if self.verbose:
                print(f'Could not find previous file: {lt_transform_fn}, computing ...')
            data = self.get_data(name)
            if not isinstance(data, da.Array):
                raise Exception('Need to provide dask array')
            lt = data.map_overlap(apply_local_thickness_dahl,
                                  depth=depth, 
                                  meta=np.array((), dtype=np.float32), 
                                  scale=scale)
            da.to_zarr(lt, lt_transform_fn, overwrite=True)

        entry = {'files': {'zarr': lt_transform_fn}, 
                 'special': special, 
                 'parent': name}
        entry = self.inherit(name, entry)
        new_name = self.add_derivative(entry)

        return new_name

    def add_manual_layer_measurement(self, index_name, cyto_name, mip_roi, 
                                     lr_index_name, rim_name, 
                                     lr_physical_voxel_size_um, 
                                     manual_layer_im_fn, 
                                     manual_layer_boundaries_fn, 
                                     manual_layer_names, pad, 
                                     special='manual.layers'):

        if index_name not in self.entries.keys():
            raise Exception(f'Entry {index_name} not found')
        if cyto_name not in self.entries.keys():
            raise Exception(f'Entry {cyto_name} not found')
        if lr_index_name not in self.entries.keys():
            raise Exception(f'Entry {lr_index_name} not found')
        if rim_name not in self.entries.keys():
            raise Exception(f'Entry {rim_name} not found')
        
        index_data = self.get_data(index_name)
        cyto_data = self.get_data(cyto_name)
        lr_index_data = self.get_data(lr_index_name)
        rim_data = self.get_data(rim_name)

        manual_layers = get_manual_layers(index_data, cyto_data, mip_roi, 
                                          lr_index_data, rim_data,
                                          lr_physical_voxel_size_um, 
                                          manual_layer_im_fn, 
                                          manual_layer_boundaries_fn, 
                                          manual_layer_names, 
                                          pad, 
                                          plot=self.verbose)
        
        self.entries[index_name]['measurements'][special] = manual_layers
        if self.verbose:
            print(f'Added measurements: {special} to entry {index_name}\n')
        return None  

    def tabulate(self, index_name, cyto_name, angio_name, physical_pixel_size_um):
        
        if index_name not in self.entries.keys():
            raise Exception(f'Entry {index_name} not found')
        if cyto_name not in self.entries.keys():
            raise Exception(f'Entry {cyto_name} not found')
        if angio_name not in self.entries.keys():
            raise Exception(f'Entry {angio_name} not found')
        
        # get cell-density info from the index-entry, and its manual.layers measurement
        manual_layers = self.get_entry(index_name)['measurements']['manual.layers']
        manual_layer_names = manual_layers['manual_layer_names']
        manual_layer_boundaries_Rdepths = manual_layers['manual_layer_boundary_Rdepths']
        manual_layer_Rdepths = manual_layers['manual_layer_Rdepths']
        manual_layer_Pdepths = manual_layers['manual_layer_Pdepths']
        manual_layer_width_Rdepths = manual_layers['manual_layer_width_Rdepths']
        manual_layer_width_Pdepths = manual_layers['manual_layer_width_Pdepths']

        table = pd.DataFrame({'layer': manual_layer_names, 
                              'relative depth [%]': np.round(100 - manual_layer_Rdepths * 100),
                              'absolute depth [mm]': np.round(manual_layer_Pdepths, 3),
                              'relative width [%]': np.round(manual_layer_width_Rdepths * 100), 
                              'absolute width [mm]': np.round(manual_layer_width_Pdepths, 3), 
                              })

        # get basic layer info from the cyto-entry, and its density.table_pp measurement
        density_table_pp = self.get_entry(cyto_name)['measurements']['density.table_pp']
        density_table_pp = pd.read_csv(density_table_pp)
        density_table_pp.index = density_table_pp.index[::-1]
        density_table_pp = density_table_pp.sort_index()
        
        density_table = pd.DataFrame({'layer': manual_layer_names, 
                              'cell density [10³ cells per mm³]': np.round(density_table_pp['density'] * (physical_pixel_size_um * 1000**3) / 10**3, 1),
                              'cell density sem [10³ cells per mm³]': np.round(density_table_pp['density_sem'] * (physical_pixel_size_um * 1000**3) / 10**3, 1),
                             })

        table = pd.merge(table, density_table, on=['layer'])

        # get vessel-orientation info from the angio-entry, and its orientation_density_table_pp measurement
        orientation_density_table_pv = self.get_entry(angio_name)['measurements']['orientation.density.table_pv']
        orientation_density_table_pv = pd.read_csv(orientation_density_table_pv)
        orientation_density_table_pv.index = orientation_density_table_pv.index[::-1]
        orientation_density_table_pv = orientation_density_table_pv.sort_index()

        percent_vasculature = np.sum(np.array(orientation_density_table_pv)[:, 1:], axis=1)
        percent_radial = orientation_density_table_pv['0'].values
        percent_tangential = orientation_density_table_pv['2']

        table['total vessel density [%]'] = np.round(percent_vasculature, 1)
        table['radial vessel density [%]'] = np.round(percent_radial, 1)
        table['tangential vessel density [%]'] = np.round(percent_tangential, 1)

        return table
        

    def measure_density(self, data_name, index_name, depthbins, databins=None, pre_chunk=None, pre_slice=None, hsv_data=False, 
                        wrap_orientation=False, mask_name=None, 
                        binary_data=True, groupby_dim=None, special='density'):
        """Measure density of data along index.
        Index provides a coordinate system, 
        depthbins are the bins in which the index and data are divided, 
        data is the thing whose density is measured.

        Optionally, a mask can be provided to only include data INSIDE the mask.

        TODO not true anymore (see databins):
        Data can be binary (then the ratio of foreground to background is computed), or
        continuous (then we count the number of voxels for each unique value at a given numerical precision (TODO currently np.round(data, 3)).

        The measurement is performed in two steps:
            - compute density
            - postprocess density
        Both steps are saved to file and paths to the .csv files are added to the measurement field of the data entry.
        NOTE: pre_chunk and pre_slice are applied before computing density, 
        and can be used to change the interpretation of the standard-error-of-the-mean (sem), see postprocess_density.

        Parameters
        ----------
        data_name : str
            name of the data entry
        index_name : str
            name of the index entry
        depthbins : int
            bin-edges applied to index
        databins : np.array
            bin-edges applied to data
        pre_chunk : list of 3 int
            chunk size of data and index
            applied before computing density
        pre_slice : tuple of 3 slice
            slice of data and index
            applied before computing density
        hsv_data : bool
            whether the data is hsv
            if True, only the first dimension of the last axis (h:orientation, from 0-to-1) is used
        wrap_orientation : bool
            whether to wrap orientation
            if True, the orientation is wrapped around the vertical axis (around .5)
        mask_name : str
            name of the mask entry
            if provided, only data inside the mask is included
        binary_data : bool
            whether the data is binary
            if True, foreground and background are computed
            if False, histogramming is done for each bin in databins
        groupby_dim : int or None
            if not None, groupby_dim indicates a dimension of the data
            along this dimension, 3D chunks are effectively subdivided into 2D slices (see indexed_hist)
            e.g. if chunks are of shape (10, 10, 10) and groupby_dim=2
            then each chunk is treated as 10x10 2D-chunks
            this might be efficient in some cases because it allows passing of 
            large 3D-chunks, while allowing to compute standard errors based their 2D-slices
            NOTE: here we assume that chunking and grouping are along the same dimension
            NOTE: e.g. 
            data is shape (1000, 2000, 2000), 
            chunks are (10, 2000, 2000), 
            groupby_dim=0
            meaning we process 100 chunks but standard errors are computed based on 1000 slices
            NOTE: DO NOT USE THIS UNLESS YOU KNOW WHAT YOU ARE DOING
        special : str
            special identifier for this derivative

        Returns
        -------
        density : pd.DataFrame
            density of data along index

        """
        if groupby_dim is not None:
            print('WARNING: groupby_dim is not None, this is an experimental feature, use with caution')

        if data_name not in self.entries.keys():
            raise Exception(f'Entry {data_name} not found')
        if index_name not in self.entries.keys():
            raise Exception(f'Entry {index_name} not found')
        if mask_name is not None:
            if mask_name not in self.entries.keys():
                raise Exception(f'Entry {mask_name} not found')
            
        out_fn = '_'.join([data_name, special + '.csv'])
        out_pp_fn = '_'.join([data_name, special + '_pp.csv'])
        out_pv_fn = '_'.join([data_name, special + '_pv.csv'])
        out_pc_fn = '_'.join([data_name, special + '_pc.csv'])
        if os.path.exists(out_fn):
            if self.verbose:
                print(f'Found previous file: {out_fn}')
            density = pd.read_csv(out_fn)
        else:
            if self.verbose:
                print(f'Could not find previous file: {out_fn}, computing ...')

            data = self.get_data(data_name)
            if hsv_data:
                if self.verbose:
                    print('Assuming hsv data: extracting h (the first dim along the last axis) ...')
                data = data[..., 0]

            if wrap_orientation:
                if self.verbose:
                    print('Wrapping orientation: moving the discontinuity from vertical (0) to horizontal (.5) ...')
                data = .5 - np.abs(data - .5)

            if mask_name is not None:
                mask = self.get_data(mask_name)
                # NOTE: for convenience, the mask is passed on in the form of a masked array 
                # (of the data)
                data = da.ma.masked_array(data, mask=mask)

            index = self.get_data(index_name)

            if pre_chunk is not None:
                data = data.rechunk(pre_chunk)
                index = index.rechunk(pre_chunk)

            if pre_slice is not None:
                data = data[pre_slice]
                index = index[pre_slice]

            if pre_chunk is not None or pre_slice is not None:
                if self.verbose:
                    print('Data after pre_chunk and/or pre_slice:')
                    display(data)
                    print('Index after pre_chunk and/or pre_slice:')
                    display(index)

            # NOTE: indexbin_volume calculates the (univariate) volume of each bin in the index
            # this can be redundant with the information in the density
            # TODO: remove redundancy (e.g. in binary mode we could just sum the counts)
            # TODO: indexbin_volume does not take into account the mask
            # TODO: allow a seperate mask for the index
            indexbin_volume = measure_hist(index, depthbins)
            indexbin_volume = indexbin_volume.rename(columns={'count': 'indexbin_volume'})
            density = measure_density(data, index, depthbins, binary_data=binary_data, 
                                      databins=databins, groupby_dim=groupby_dim)
            if groupby_dim is not None:
                # NOTE: this is a bit hacky, but it works
                chunk_id = density['chunk_id'].values * density['groupby_dim'].max().values + density['groupby_dim'].values
                chunk_id = np.array(chunk_id).astype(int)
                density['chunk_id'] = chunk_id

            density = pd.merge(indexbin_volume, density, on=['index', 'chunk_id'])
            # NOTE: the problem with merging like this is that 
            # 'indexbin_volume' gets repeated for each 'value'-row in a 'index'x'chunk_id' group
            density.to_csv(out_fn)
        self.entries[data_name]['measurements'][special] = out_fn
        if self.verbose:
            print(f'Added measurements: {special} to entry {data_name}\n')

        # TODO adjust postprocessing for continuous data
        if os.path.exists(out_pp_fn) and os.path.exists(out_pv_fn) and os.path.exists(out_pc_fn):
            if self.verbose:
                print(f'Found previous file: {out_pp_fn} and {out_pv_fn} and {out_pc_fn}')
            density_pp = pd.read_csv(out_pp_fn)
            density_pv = pd.read_csv(out_pv_fn)
            density_pc = pd.read_csv(out_pc_fn)
        else:
            if self.verbose:
                print(f'Could not find previous file: {out_pp_fn}, computing ...')
            # pp postprocessed, pv per_value, and pc per_chunk
            density_pp, density_pv, density_pc = postprocess_density(density, depthbins, binary_data=binary_data, databins=databins)
            density_pp.to_csv(out_pp_fn)
            density_pv.to_csv(out_pv_fn)
            density_pc.to_csv(out_pc_fn)
        self.entries[data_name]['measurements'][special + '_pp'] = out_pp_fn
        self.entries[data_name]['measurements'][special + '_pv'] = out_pv_fn
        self.entries[data_name]['measurements'][special + '_pc'] = out_pc_fn
        if self.verbose:
            print(f'Added measurements: {special}_pp to entry {data_name}\n')
            print(f'Added measurements: {special}_pv to entry {data_name}\n')
            print(f'Added measurements: {special}_pc to entry {data_name}\n')
        return None
    
    def cellcenters_to_volume(self, name, mask_name=None, mask_label=False, special='cellcenters'):
        """Convert cellcenters to volume.
        Given an entry with cellsegmentation which also contains a measurement of cellcenters,
        this function generates a volume from the cellcenters.
        Optionally, a mask can be provided to only include cellcenters OUTSIDE the mask.
        Optionally, the original label can be adjusted to only contain cells whose cellcenters fall OUTSIDE the mask.
        """
        if name not in self.entries.keys():
            raise Exception(f'Entry {name} not found')
        if mask_name is not None:
            if mask_name not in self.entries.keys():
                raise Exception(f'Entry {mask_name} not found')
        
        has_cellcenters = False
        if 'measurements' in self.entries[name].keys():
            if 'points' in self.entries[name]['measurements'].keys():
                cellcenters = self.entries[name]['measurements']['points']
                has_cellcenters = True
        if not has_cellcenters:
            raise Exception(f'Need to provide cellcenters for entry {name}')

        cellcenters_volume_fn = '_'.join([name, special + '.zarr'])
        masked_label_fn = '_'.join([name, special + '.masked.label.zarr'])

        if os.path.exists(cellcenters_volume_fn):
            if self.verbose:
                print(f'Found previous files: {cellcenters_volume_fn}')
                if mask_label:
                    if os.path.exists(masked_label_fn):
                        print(f'Found previous files: {masked_label_fn}')
        else:
            if self.verbose:
                print(f'Could not find previous file: {cellcenters_volume_fn}, computing ...')

            data = self.get_data(name)
            if mask_name is not None:
                mask = self.get_data(mask_name)
            cellcenters_outside_mask = volume_from_cellcenters(cellcenters, data, cellcenters_volume_fn, mask=mask)

            if mask_label:
                if mask_name is None:
                    raise Exception('Need to provide mask_name to adjust label')
                if self.verbose:
                    print(f'Masking the labels and generating {masked_label_fn}, computing ...')
                data_zarr_fn = self.get_entry(name)['files']['zarr']
                data_zarr = zarr.open(data_zarr_fn)
                include = data_zarr[cellcenters_outside_mask[:, 0], cellcenters_outside_mask[:, 1], cellcenters_outside_mask[:, 2]]
                masked_label = data.map_blocks(accept_labels, 
                                               dtype=data.dtype, include=include)
                da.to_zarr(masked_label, masked_label_fn)

        entry = {'files': {'zarr': cellcenters_volume_fn}, 
                           'special': special, 
                           'parent': name}
        entry = self.inherit(name, entry)
        new_name = self.add_derivative(entry)

        if mask_label:
            entry = {'files': {'zarr': masked_label_fn}, 
                               'special': special + '.masked.label', 
                               'parent': name}
            entry = self.inherit(name, entry)
            new_name = self.add_derivative(entry)

        return new_name


    def cellcenters_to_areas(self, cellcenters_name, label_name, special='cellcenters.areas'):
        """Measure areas of cellcenters.
        Given an entry with cellcenters and an entry with a label, 
        this function measures the area of each cellcenter in the label.
        """
        if cellcenters_name not in self.entries.keys():
            raise Exception(f'Entry {cellcenters_name} not found')
        if label_name not in self.entries.keys():
            raise Exception(f'Entry {label_name} not found')
        
        cellcenters = self.get_data(cellcenters_name)
        label = self.get_data(label_name)
        
        labels = label[cellcenters].compute()
        
        regionprops = apply_chunked_regionprops(label)
        regionprops = regionprops.loc[regionprops['label'].isin(labels)]
        
        areas = regionprops['area'].values
        labels = regionprops['label'].values

        xsorted = np.argsort(labels)
        res = xsorted[np.searchsorted(labels[xsorted], labels)]
        areas_sorted = areas[res]
        areas_sorted = areas_sorted.astype(np.uint16)

        cellcenters_areas = da.zeros_like(cellcenters).astype(areas_sorted.dtype)
        cellcenters_areas[cellcenters] = areas_sorted

        out_fn = '_'.join([cellcenters_name, special + '.csv'])
        da.to_zarr(cellcenters_areas, out_fn, overwrite=True)

        entry = {'files': {'zarr': out_fn},
                           'special': special,
                           'parent': cellcenters_name}
        entry = self.inherit(cellcenters_name, entry)
        new_name = self.add_derivative(entry)

        return new_name







            


        
        