import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import seaborn as sns

import dask
import dask.array as da

from skimage.exposure import match_histograms

from microfilm.microplot import microshow
from microfilm import microplot
from microfilm.microplot import Micropanel

from .filters import preproc_angio_for_vis, preproc_cyto_for_vis
from .core import get_avg_bin_loc


def depthplot(angio_density_pp=None, angio_density_pc=None, cyto_density_pp=None, cyto_density_pc=None):
    # TODO: remove this function    
    raise NotImplementedError('This function is not yet implemented')
    if np.all([angio_density_pp is None, 
               angio_density_pc is None,
               cyto_density_pp is None, 
               cyto_density_pc is None]):
        raise Exception('Need to pass some data')
        
    fig, ax = plt.subplots(2, 2, figsize=(5.4, 12.8), sharey=False, dpi=700)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=None)
    ax = ax.ravel()
    
    purple = pdm_v.purple_lut()
    green = pdm_v.green_lut()
    
    # ROI volume
    if angio_density_pp is not None:
        ax[0].plot(angio_density_pp['percent_indexbin_volume'],
                   angio_density_pp['index_bin'], 
                   c='k', 
                   linestyle='-', 
                   label='V1', 
                   linewidth=.5)
        ax[0].set_xlabel('ROI volume (% of cortex)')
    else:
        ax[0].plot(cyto_density_pp['percent_indexbin_volume'],
                   cyto_density_pp['index_bin'], 
                   c='k', 
                   linestyle='-', 
                   label='V1', 
                   linewidth=.5)
        ax[0].set_xlabel('ROI volume (% of cortex)')
    
    # Angio
    if angio_density_pp is not None:
        l1 = ax[1].plot(angio_density_pp['percent_density'], 
                        angio_density_pp['index_bin'], 
                        c=purple(100), 
                        linewidth=.5, 
                        linestyle='-', 
                        label='V1')
        ax[1].fill_betweenx(angio_density_pp['index_bin'], 
                            angio_density_pp['percent_density'] + 2*angio_density_pp['percent_density_sem'], 
                            angio_density_pp['percent_density'] - 2*angio_density_pp['percent_density_sem'], 
                            color=purple(100), 
                            alpha=.2)
        ax[1].set_xlim(2, 16)
        ax[1].spines['bottom'].set_color(purple(100))
        ax[1].set_xlabel('vessel density (%)')
    
    # Cyto
    if cyto_density_pp is not None:
        ax1 = ax[1].twiny()
        ax1.set_xlabel('cell density (10³ cells per mm³)')
        ax1.plot(cyto_density_pp['percent_density'], 
                 cyto_density_pp['index_bin'], 
                 c=green(100), 
                 linewidth=.5)
        ax1.fill_betweenx(cyto_density_pp['index_bin'], 
                          cyto_density_pp['percent_density'] + 2*cyto_density_pp['percent_density_sem'], 
                          cyto_density_pp['percent_density'] - 2*cyto_density_pp['percent_density_sem'], 
                          color=green(100), 
                          alpha=.2)
        ax1.set_xticks([25000, 50000, 75000, 100000])
        ax1.set_xticklabels(['25', '50', '75', '100'])
        ax1.spines['top'].set_color(green(100))
        ax1.spines['bottom'].set_color(purple(100))
    
    
    
    ax[0].spines[['right']].set_visible(False)
    ax[0].set_yticks([0, 100])
    ax[0].set_yticklabels(['WM', 'CSF'])
    
    ax[1].spines[['right']].set_visible(False)
    #ax1.spines[['right']].set_visible(False)
    ax[1].set_yticks([0, 100])
    ax[1].set_yticklabels(['WM', 'CSF'])
    
    # peak ticks
    #ax[1].set_yticks(p)
    #ax[1].set_yticklabels(labels)
    
    #for xtick, color in zip(ax[1].get_yticklabels(), c):
    #    xtick.set_color(color)
    #ax[1].set_ylabel('cortical depth (mm)')
    
    if angio_density_pc is not None:
        l1 = ax[2].plot(np.array(angio_density_pc), 
                        angio_density_pc['index'], 
                        alpha=.002, 
                        c=purple(100), 
                        label='V1', 
                        zorder=1)
        
    if cyto_density_pc is not None:
        l1 = ax[3].plot(np.array(cyto_density_pc), 
                        cyto_density_pc['index'], 
                        alpha=.002, 
                        c=purple(100), 
                        label='V1', 
                        zorder=1)

    ax[2].set_xlim(0, 22)
    ax[2].spines['bottom'].set_color(purple(100))
    ax[2].set_xlabel('vessel density (%)')
    

def cross_area_plot(area1_df, area2_df, 
                    area_names=['area1', 'area2'],
                    mode=['cyto', 'angio'][0], 
                    scale_sem=1,
                    physical_pixel_size_um=(1.03 * 1.03 * 1), 
                    show_legend=False):

    depth = area1_df['index'].values / np.max(area1_df['index'])

    fig, ax = plt.subplots(1, 1, figsize=(3, 10))

    area_colors = [(1, 0, 0, 1), (0, 0, 1, 1)]

    for area_name, area_color, area_df in zip(area_names, 
                                              area_colors, 
                                              [area1_df, area2_df]):
        if mode == 'angio':

            l1 = ax.plot(area_df['percent_density'], 
                            depth, 
                            c=area_color, 
                            linewidth=.5, 
                            linestyle='-')
            p1 = ax.fill_betweenx(depth, 
                                area_df['percent_density'] + scale_sem * area_df['percent_density_sem'], 
                                area_df['percent_density'] - scale_sem * area_df['percent_density_sem'], 
                                color=area_color, #np.minimum(1, np.array(angio_color)+.5),  
                                alpha=.2, 
                                label=area_name)
            ax.set_xlim(2, 16)
            #ax.spines['bottom'].set_color(area_color)
            ax.set_xlabel('Vessel density [%]')
            #ax.xaxis.label.set_color(area_color)

        elif mode == 'cyto':
            # NOTE: conversion from density to number of cells per mm3
            n_cells_per_mm3 = area_df['density'] * (physical_pixel_size_um * 1000**3)
            sem = area_df['density_sem'] * (physical_pixel_size_um * 1000**3)
            
            ax1 = ax
            ax1.set_xlabel('Cell density [10³ cells per mm³]')
            #ax1.xaxis.label.set_color(area_color)
            l1 = ax1.plot(n_cells_per_mm3, 
                    depth,
                    c=area_color, 
                    linewidth=.5)
            p1 = ax1.fill_betweenx(depth, 
                            n_cells_per_mm3 + scale_sem * sem, 
                            n_cells_per_mm3 - scale_sem * sem, 
                            color=area_color, 
                            alpha=.2, 
                            label=area_name)
            ax1.set_xticks([25_000, 50_000, 75_000, 100_000])
            ax1.set_xticklabels(['25', '50', '75', '100'])
            #ax1.spines['top'].set_color(area_color)
            #ax1.spines['bottom'].set_color(area_color)

        if show_legend:
            ax.legend(loc='upper right')
        ax.set_yticks([1, 0])
        ax.set_yticklabels(['ps', 'wm'])

    return fig



def stacked_depthplot_density(df, valuebins_edges, normalize=False, cmap='gnuplot2', bins=None, colors=None, legend=False):
    
    depth = df['index'].values / np.max(df['index'])
    data = np.array(df)[:, 1:]
    depthbins, valuebins = data.shape
    
    if bins is not None:
        #orientation_bins = np.array([0, 10, 80, 90])
        #orientation_bins = orientation_bins / 180
        depths = get_avg_bin_loc(valuebins_edges)
        bins = np.digitize(depths, bins) -1
        new_valuebins_edges = np.zeros(len(np.unique(bins))+1)
#        print(bins, len(np.unique(bins)))
        data_ = [np.zeros(depthbins) for i in range(len(np.unique(bins)))]
        for i, col in enumerate(data.T):
            bin_i = bins[i]
            data_[bin_i] += col
            value_bin_l = valuebins_edges[i]
            value_bin_r = valuebins_edges[i+1]
            new_valuebins_edges[bin_i] = np.minimum(value_bin_l, new_valuebins_edges[bin_i])
            new_valuebins_edges[bin_i+1] = np.maximum(value_bin_r, new_valuebins_edges[bin_i+1])
            
        data = np.array(data_).T
    
    if normalize:
        data = data / np.sum(data, axis=1)[:, np.newaxis]
    
    if bins is not None:
        C = sns.color_palette(cmap, len(np.unique(bins)))
    else:
        C = sns.color_palette(cmap, valuebins)

    fig, ax = plt.subplots(1, 1, figsize=(3, 10))

    start = np.zeros(depthbins)
    for i, dp in enumerate(np.cumsum(data, axis=1).T):

        if colors is not None:
            c = colors[i]
        else:
            c = C[i]
        ax.fill_betweenx(depth, 
                          dp, 
                          start, 
                          color=c, 
                          alpha=.4)


        if bins is not None:
            label = f'{np.round(new_valuebins_edges[i], 2)} - {np.round(new_valuebins_edges[i+1])}' + u' \u03bcm'
        else:
            label = f'{np.round(valuebins_edges[i], 2)} - {np.round(valuebins_edges[i+1])}' + u' \u03bcm'
        ax.plot(dp, 
                 depth, 
                 alpha=.5, 
                 c=c, 
                 #linewidth=bin_edges[i+1], 
                 label=label)

        start = dp
    
    if legend:
        ax.legend(bbox_to_anchor=(1.6, 1.01), loc="upper right", title="vessel radius")

    ax.set_yticks([1, 0])
    ax.set_yticklabels(['ps', 'wm'])
    ax.set_ylabel('Cortical depth')
    ax.spines[['right', 'top']].set_visible(False)
    
    if normalize:
        ax.set_xticks([0, .2, .4, .6, .8, 1])
        for i in [0, .2, .4, .6, .8, 1]:
            ax.axvline(i, c='k', alpha=.5, linewidth=.5)
        ax.set_xticklabels(np.arange(0, 101, 20))
        ax.set_xlabel('Vessel density [% of vasculature]')
    else:
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax.set_xlabel('Vessel density [% of vasculature]')

    #plt.savefig('plots/v1_vessel_thickness_cumulative.png', bbox_inches="tight")
    #plt.savefig('plots/v1_vessel_thickness_cumulative.svg', bbox_inches="tight")
    return fig


def plot_layer_bars(manual_layer_boundaries, ax, manual_layer_names=None, add_layer_ticks=False):
    
    manual_layer_centers = []
    manual_layer_heights = []
    manual_layer_colors = []
    for i in range(len(manual_layer_boundaries)-1):
        manual_layer_centers.append(manual_layer_boundaries[i+1] + ((manual_layer_boundaries[i] - manual_layer_boundaries[i+1])/2))
        manual_layer_heights.append(manual_layer_boundaries[i] - manual_layer_boundaries[i+1])
        manual_layer_colors.append(['k', 'w'][i%2])

    xlim = ax.get_xlim()
    rects = ax.barh(manual_layer_centers, 
                    xlim[1]*1, 
                    align='center', 
                    height=manual_layer_heights, 
                    color=manual_layer_colors,
                    alpha=.2)
    ax.set_xlim(xlim)

    if manual_layer_names is not None:
        for p, txt in zip(manual_layer_centers, manual_layer_names):
            t = ax.text(2.25, p, txt, ha='left', va='center')

    if isinstance(add_layer_ticks, bool):
        if add_layer_ticks:
            ax.set_yticks(np.round(manual_layer_boundaries, 2))
    else:
        if isinstance(add_layer_ticks, list) or isinstance(add_layer_ticks, np.ndarray):
            ax.set_yticks(np.round(manual_layer_boundaries, 2))
            ax.set_yticklabels(add_layer_ticks)   
    
    return None

    
def depthplot_density(angio_pp=None, cyto_pp=None, angio_pc=None, cyto_pc=None, 
                      angio_color=None, cyto_color=None, peak_df=None, 
                      physical_pixel_size_um=(1.03 * 1.03 * 1), 
                      scale_sem=1, 
                      manual_layer_boundaries=None,
                      manual_layer_names=None,
                      add_layer_ticks=None):
    """
    # TODO: maybe parameterize axis limits
    """

    # normalize
    if angio_pp is not None and cyto_pp is not None:
        max_indexbin = max(angio_pp['index_bin'].max(), cyto_pp['index_bin'].max())
        angio_pp['index_bin'] = angio_pp['index_bin'] / max_indexbin
        cyto_pp['index_bin'] = cyto_pp['index_bin'] / max_indexbin
    elif angio_pp is not None:
        max_indexbin = angio_pp['index_bin'].max()
        angio_pp['index_bin'] = angio_pp['index_bin'] / max_indexbin
    elif cyto_pp is not None:
        max_indexbin = cyto_pp['index_bin'].max()
        cyto_pp['index_bin'] = cyto_pp['index_bin'] / max_indexbin

    
    if np.all([angio_pp is None, 
               angio_pc is None,
               cyto_pp is None, 
               cyto_pc is None]):
        raise Exception('Need to pass some data')
    
    if angio_color is None:
        angio_color = purple_lut()(100)
        angio_txt_color = purple_lut()(75)
    if cyto_color is None:
        cyto_color = green_lut()(100)
        cyto_txt_color = green_lut()(75)


    fig, ax = plt.subplots(2, 2, figsize=(5.4, 12.8), sharey=False, dpi=700)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=None)
    ax = ax.ravel() 

    if angio_pp is not None:
        depth = angio_pp['index_bin'].values#len(angio_pp)
    else:
        depth = cyto_pp['index_bin'].values
    
    # ROI/layer volume
    if angio_pp is not None:
        l1 = ax[0].plot(angio_pp['percent_indexbin_volume'], 
                        angio_pp['index_bin'], 
                        c='k', 
                        linestyle='-', 
                        label='V1', 
                        linewidth=.5)
        ax[0].set_xlabel('ROI volume (% of cortex)')
    else:
        l1 = ax[0].plot(cyto_pp['percent_indexbin_volume'], 
                        cyto_pp['index_bin'], 
                        c='k', 
                        linestyle='-', 
                        label='V1', 
                        linewidth=.5)
        ax[0].set_xlabel('ROI volume (% of cortex)')
        
    # angio
    if angio_pp is not None:
        l1 = ax[1].plot(angio_pp['percent_density'], 
                        angio_pp['index_bin'], 
                        c=angio_color, 
                        linewidth=.5, 
                        linestyle='-', 
                        label='V1')
        ax[1].fill_betweenx(angio_pp['index_bin'], 
                            angio_pp['percent_density'] + scale_sem * angio_pp['percent_density_sem'], 
                            angio_pp['percent_density'] - scale_sem * angio_pp['percent_density_sem'], 
                            color=angio_color, #np.minimum(1, np.array(angio_color)+.5),  
                            alpha=.2)
        ax[1].set_xlim(2, 16)
        ax[1].spines['bottom'].set_color(angio_color)
        ax[1].set_xlabel('Vessel density [%]')
        ax[1].xaxis.label.set_color(angio_txt_color)


    # cyto
    if cyto_pp is not None:
        # NOTE: conversion from density to number of cells per mm3
        n_cells_per_mm3 = cyto_pp['density'] * (physical_pixel_size_um * 1000**3)
        sem = cyto_pp['density_sem'] * (physical_pixel_size_um * 1000**3)
        
        ax1 = ax[1].twiny()
        ax1.set_xlabel('Cell density [10³ cells per mm³]')
        ax1.xaxis.label.set_color(cyto_txt_color)
        ax1.plot(n_cells_per_mm3, 
                 cyto_pp['index_bin'],
                 c=cyto_color, 
                 linewidth=.5)
        ax1.fill_betweenx(depth, 
                          n_cells_per_mm3 + scale_sem * sem, 
                          n_cells_per_mm3 - scale_sem * sem, 
                          color=cyto_color, 
                          alpha=.2)
        ax1.set_xticks([25_000, 50_000, 75_000, 100_000])
        ax1.set_xticklabels(['25', '50', '75', '100'])
        ax1.spines['top'].set_color(cyto_color)
        ax1.spines['bottom'].set_color(angio_color)

    # manual layer boundaries
    if manual_layer_boundaries is not None:
        plot_layer_bars(manual_layer_boundaries, 
                        ax[1], 
                        manual_layer_names=manual_layer_names, 
                        add_layer_ticks=add_layer_ticks)

    # angio
    # NOTE: avoid index column
    if angio_pc is not None:
        l1 = ax[2].plot(np.array(angio_pc)[:, 1:], 
                        depth, 
                        alpha=.2, 
                        c=angio_color, 
                        linewidth=.5, 
                        label='V1', 
                        zorder=1)

        ax[2].set_xlim(0, 22)
        ax[2].spines['bottom'].set_color(angio_color)
        ax[2].set_xlabel('Vessel density [%]')

    # cyto
    if cyto_pc is not None:
        ax3 = ax[3]#.twiny()
        ax3.set_xlabel('Cell density [10³ cells per mm³]')
        # NOTE: avoid index column
        # NOTE: conversion from PERCENT-density to number of cells per mm3
        ax3.plot(np.array(cyto_pc)[:, 1:] / 100 * (physical_pixel_size_um * 1000**3), 
                depth, 
                alpha=.2, 
                c=cyto_color, 
                linewidth=.5);
        #ax3.set_xlim([0, 230000])
        ax3.set_xticks([100_000, 200_000])
        ax3.set_xticklabels(['100', '200'])
        ax3.spines['bottom'].set_color(cyto_color)

    ax[0].spines[['right']].set_visible(False)
    ax[0].set_yticks([depth[0], depth[-1]])
    ax[0].set_yticklabels(['wm', 'ps'])

    ax[1].spines[['right']].set_visible(False)
    if cyto_pp is not None:
        ax1.spines[['right']].set_visible(False)
    ax[1].set_yticks([depth[0], depth[-1]])
    ax[1].set_yticklabels(['wm', 'ps'])

    # peak ticks
    if peak_df is not None:
        p = np.hstack([depth[0], peak_df['peak'], depth[-1]])
        labels = np.hstack(['wm', peak_df['label'], 'ps'])
        black = np.array([0, 0, 0, 1])
        c = np.vstack([black, np.array([np.array(c) for c in peak_df['color']]), black])
        ax[1].set_yticks(p)
        ax[1].set_yticklabels(labels)

        for xtick, color in zip(ax[1].get_yticklabels(), c):
            xtick.set_color(color)
        ax[1].set_ylabel('cortical depth (mm)')
        
    ax[2].spines[['right']].set_visible(False)
    ax[2].set_yticks([depth[0], depth[-1]])
    ax[2].set_yticklabels(['wm', 'ps'])

    ax[3].spines[['right']].set_visible(False)
    ax[3].set_yticks([depth[0], depth[-1]])
    ax[3].set_yticklabels(['wm', 'ps'])

    return fig

def inspect_angio_segmentation(angio, cyto, lk, log, tm, roi, roi_mip):
    """
    """

    def standard_plot(im, 
                      limits, 
                      cmaps, 
                      label_text, 
                      channel_names):
        if len(im.shape) == 2:
            im = [im]
        microim = microplot.Microimage(images=im, 
                                       limits=limits, 
                                       proj_type='max', 
                                       cmaps=cmaps, 
                                       label_text=label_text, 
                                       label_color='white', 
                                       label_kwargs={'backgroundcolor': 'k'},
                                       channel_label_show=True, 
                                       channel_names=[channel_names],
                                       unit='um', 
                                       scalebar_unit_per_pix=1.03, 
                                       scalebar_size_in_units=50, 
                                       scalebar_thickness=.01, 
                                       scalebar_kwargs={'frameon': True, 
                                                         'box_color': 'k', 
                                                         'box_alpha': 1})
        return microim
    
    # angio (slice)
    im = angio[roi].compute().squeeze()
    im = preproc_angio_for_vis(im)
    microim1 = standard_plot(im,
                             [np.percentile(np.ravel(im), [5, 95])],
                             ['gray'],
                             'angio', 
                             ['angio'])
    
    # cyto (slice)
    im = cyto[roi].compute().squeeze()
    im = preproc_cyto_for_vis(im)
    microim2 = standard_plot(im,
                             [np.percentile(np.ravel(im), [0, 95])],
                             ['gray'],
                             'cyto', 
                             ['cyto'])
    # Labkit + TubeMap
    im = np.stack([np.squeeze(lk[roi] > 0, axis=0), 
                   np.squeeze(tm[roi] > 0, axis=0)])
    microim3 = standard_plot(im,
                             [[0, 1], [0, 1]],
                             ['pure_red', 'pure_cyan'],
                             'Labkit + TubeMap', 
                             ['Labkit', 'TubeMap'])

    # Labkit + LoG
    im = np.stack([np.squeeze(lk[roi] > 0, axis=0), 
                   np.squeeze(log[roi] > 0, axis=0)])
    microim4 = standard_plot(im,
                             [0, 1],
                             ['pure_red', 'pure_cyan'],
                             'Labkit + LoG', 
                             ['Labkit', 'LoG'])

    # angio (max IP)
    im = angio[roi_mip].compute()
    im = np.max(im, axis=0)
    im = preproc_angio_for_vis(im)
    microim5 = standard_plot(im,
                             [np.percentile(np.ravel(im), [5, 95])],
                             ['gray'],
                             'angio (MaxIP)', 
                             ['angio'])

    # cyto (min IP)
    im = cyto[roi_mip].compute()
    im = np.min(im, axis=0)
    #im = np.percentile(im, 10, axis=0)
    im = preproc_cyto_for_vis(im)
    microim6 = standard_plot(im,
                                [np.percentile(np.ravel(im), [0, 95])],
                                ['gray'],
                                'cyto (minIP)', 
                                ['cyto'])

    # Labkit + TubeMap
    im = np.stack([np.max(lk[roi_mip] > 0, axis=0), 
                   np.max(tm[roi_mip] > 0, axis=0)])
    microim7 = standard_plot(im,
                             [0, 1],
                             ['pure_red', 'pure_cyan'],
                             'Labkit + TubeMap', 
                             ['Labkit', 'TubeMap'])

    # Labkit + LoG
    im = np.stack([np.max(lk[roi_mip] > 0, axis=0), 
                   np.max(log[roi_mip] > 0, axis=0)])
    microim8 = standard_plot(im,
                             [0, 1],
                             ['pure_red', 'pure_cyan'],
                             'Labkit + LoG', 
                             ['Labkit', 'LoG'])

    panel = Micropanel(rows=4, cols=2, figsize=[10, 10], figscaling=3)
    panel.add_element([0, 0], microim1);
    panel.add_element([0, 1], microim2);
    panel.add_element([1, 0], microim3);
    panel.add_element([1, 1], microim4);
    panel.add_element([2, 0], microim5);
    panel.add_element([2, 1], microim6);
    panel.add_element([3, 0], microim7);
    panel.add_element([3, 1], microim8);

    return panel
    

def plot_extent_vs_area(ax, df, label, areadotsize=False, extent_in_cube=True, cumulative=True):
    """
    Plot extent vs area for a dataframe

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        matplotlib axes object
    df : pd.DataFrame
        dataframe with columns 'area' and 'extent'
    label : str
        label for the legend
    areadotsize : bool, optional
        if True, the size of the dots is proportional to the area, by default False
    cumulative : bool, optional
        if True, a cumulative area plot is added to the right y-axis, by default True
    """

    # Plot cumulative first
    # Then, in case we need to plot a subset of the data, we still get an accurate cumulative plot
    if cumulative:
        ax2 = ax.twinx()
        df.sort_values(by=['area'], ascending=True, inplace=True)
        ax2.step(df['area'], np.cumsum(df['area'])/np.sum(df['area']), c='r', where='post', zorder=1)
        ax2.set_ylabel('cumulative area (not subsampled)', color='r')
        ax2.set_yticks([])
    
    if areadotsize:
        if df.size > 20000:
            warnings.warn(f'df has too many elements for scattering with areadotsize=True, trying to plot a subsample of the data ...')
            df.sort_values(by=['area'], ascending=False, inplace=True)
            # select the largest 10 and then every 100th value for plotting
            df = pd.concat([df[:10], df[10::100]])
        s = df['area']/df['area'].sum()
        # TODO: parameterize this
        # TODO: area-to-dotsize scaling linear/logarithmic? in terms of diameter or area?
        s *= 100 
        s += 1
        alpha = .25
        label += ' (subsampled)'
    else:
        s = .1
        alpha=.1

    def get_cubic_extent(df):
        longest_side = np.max(np.stack([df['x_stop'] - df['x_start'], 
                                        df['y_stop'] - df['y_start'], 
                                        df['z_stop'] - df['z_start']]), axis=0)
        cubic_extent = longest_side / df['area']
        return cubic_extent
    if extent_in_cube:
        df['extent'] = get_cubic_extent(df)        

    ax.scatter(df['area'], 
               df['extent'], 
               alpha=alpha, 
               c='k', 
               s=s, 
               label=label, 
               zorder=0)

    ax.set_xscale('log')
    ax.set_xlabel('area')
    if extent_in_cube:
        ax.set_ylabel('extent in cube')
    else:
        ax.set_ylabel('extent')
    ax.legend(loc='upper right')

    return None


def plot_upsampled_layers(metric, depthbins, roi=None):

    roi = get_2Dslice_in3D(roi, metric)
    metric = metric[roi].squeeze().T

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    alpha = np.logical_and(metric<1, metric>0).astype(float)
    ax[0].imshow(metric, alpha=alpha, interpolation='none')
    ax[0].axis('off')
    plot_layers(metric, depthbins, ax=ax[1], alpha=alpha)
    ax[1].axis('off')

def plot_layers(index, depthbins, ax=None, alpha=None, cmap=None):

    if np.any(np.diff(depthbins) > 0):
        raise ValueError('depthbins must be in decending order')
    im = np.zeros_like(index, dtype=float)
    start = depthbins[0]
    for i, stop in enumerate(depthbins[1:]):
        im[np.logical_and(index < start,
                          index >= stop)] = 1 + (i % 2)
        start = stop
    im[im == 0] = np.nan
    if ax:
        if not alpha is None:
            ax.imshow(im, alpha=alpha, cmap=cmap, vmin=0)
        else:
            ax.imshow(im, cmap=cmap)
    else:
        if not alpha is None:
            plt.imshow(im, alpha=alpha, cmap=cmap, vmin=0)
        else:
            plt.imshow(im, cmap=cmap)

def get_2Dslice_in3D(roi, data):
    """Get a roi that gets a 2D slice from 3D data.
    e.g for visualizations

    Allows some convenient shorthands:
    e.g.
    [100, None, None] ->
    (slice(100, 101, None), slice(None, None, None), slice(None, None, None))
    or
    [100, [10, 40], None] ->
    (slice(100, 101, None), slice(10, 40, None), slice(None, None, None))
    
    and, defaults to middle slice of last dimension

    Parameters
    ----------
    roi : list
        list of 3 ints or lists
    data : np.ndarray
        3D data

    Returns
    -------
    ROI : tuple
        tuple of 3 slices
    """


    # default to middle slice of last dimension
    if roi is None:
        roi = [None, None, int(data.shape[2]/2)]

    if len(roi) != 3:
        raise ValueError('roi must have 3 dimensions')

    # if list of int/None, assert 2 dimensions are None
    if isinstance(roi, list):
        if np.all([isinstance(dim, int) or dim is None for dim in roi]):
            assert np.sum([isinstance(r, int) for r in roi]) == 1
            assert np.sum([r is None for r in roi]) == 2
    # convert to tuple of slices
    ROI = []
    for idim, dim in enumerate(roi):
        # int to list
        if isinstance(dim, int):
            dim = [dim, dim+1, None]
        elif dim is None:
            dim = [None, None, None]
        # list to slice
        if isinstance(dim, list):
            if len(dim) == 1:
                slc = slice(dim[0], dim[0]+1, None)
            elif len(dim) == 2:
                slc = slice(dim[0], dim[1], None)
            else:
                slc = slice(dim[0], dim[1], dim[2])
        elif isinstance(dim, slice):
            slc = dim
        else:
            raise ValueError('roi must consist of ints, lists, or slices')
        ROI.append(slc)
    ROI = tuple(ROI)
    return ROI

def get_ChrisLUTs():
    # unused because microfilm has these build in
    ## Colormaps (for channel overlays)
    ## from https://forum.image.sc/t/non-rgb-channel-image-lut-overlay-visualisation-in-python/67242/5
    ## A helper function to pull in some custom LUTs:
    def get_lut(raw_url):
        """Download an ImageJ formatted (?) LUT and build MPL colormap"""
        ds = np.DataSource(None)  # needed for the numpy open method(?)
        fixed_url = raw_url.replace(" ", "%20")  # spaces in URLs not cool
        # Download the LUT file, trim parts of the table, convert to fraction:
        lut_rgb = (np.loadtxt(ds.open(fixed_url), skiprows=1)[:, -3:]) / 256
        return lut_rgb
    # URLs for some excellent Blue-Orange-Purple LUTs
    bop_urls = (
        "https://raw.githubusercontent.com/cleterrier/ChrisLUTs/master/BOP blue.lut",
        "https://raw.githubusercontent.com/cleterrier/ChrisLUTs/master/BOP orange.lut",
        "https://raw.githubusercontent.com/cleterrier/ChrisLUTs/master/BOP purple.lut",
        "https://raw.githubusercontent.com/cleterrier/ChrisLUTs/master/I Cyan.lut",
    )
    # Download the LUTs and convert to matplotlib colormap format
    bop_cmaps = {}
    for url in bop_urls:
        bop_cmaps[url.rsplit('/', 1)[-1]].append(ListedColormap(get_lut(url)))
    return bop_cmaps    # bop_cmaps['BOP blue.lut'] etc.

def cyan_lut():
    lut = np.zeros((100, 3))
    lut[:, 0] = 0                       # r = 0
    lut[:, 1] = np.linspace(0, 1, 100)  # g = [0, 1]
    lut[:, 2] = np.linspace(0, 1, 100)  # b = [0, 1]
    return ListedColormap(lut)

def purple_lut():
    lut = np.zeros((100, 3))
    lut[:, 0] = np.linspace(0, 1, 100)  # r = [0, 1]
    lut[:, 1] = 0                       # g = 0
    lut[:, 2] = np.linspace(0, 1, 100)  # b = [0, 1]
    return ListedColormap(lut)

def green_lut():
    lut = np.zeros((100, 3))
    lut[:, 0] = 0                       # r = [0, 1]
    lut[:, 1] = np.linspace(0, 1, 100)  # g = 0
    lut[:, 2] = 0                       # b = [0, 1]
    return ListedColormap(lut)


def show_before_after(before, after, cmap, roi=None, mip=None, 
                      preprocess=None, 
                      match_histograms_between_channels=True, 
                      scalebar_um_per_pix=1.03,
                      scalebar_size_in_units=50):
    if not isinstance(before, list):
        before = [before]
    if not isinstance(after, list):
        after = [after]
    if not isinstance(cmap, list):
        cmap = [cmap]

    if roi is not None:
        before = [b[roi].squeeze() for b in before]
        after = [a[roi].squeeze() for a in after]
    if mip is not None:
        before = [np.max(b, axis=mip) for b in before]
        after = [np.max(a, axis=mip) for a in after]
    if isinstance(before[0], da.Array):
        before = [b.compute() for b in before]
    if isinstance(after[0], da.Array):
        after = [a.compute() for a in after]
    if preprocess is not None:
        before = [pp(b) for pp, b in zip(preprocess, before)]
        after = [pp(a) for pp, a in zip(preprocess, after)]
    if match_histograms_between_channels:
        before = [b if i==0 else match_histograms(b, before[0]) for i, b in enumerate(before)]
        after = [a if i==0 else match_histograms(a, after[0]) for i, a in enumerate(after)]

    limits = [[np.percentile(b, 1), np.percentile(b, 99)] for b in before]
    im = np.stack(before)        
    microim1 = microplot.Microimage(images=im, 
                                    proj_type='sum', 
                                    cmaps=cmap, 
                                    rescale_type='limits',
                                    limits=limits, 
                                    label_text='', 
                                    label_color='white', 
                                    channel_label_show=True, 
                                    unit='um', 
                                    scalebar_unit_per_pix=scalebar_um_per_pix, 
                                    scalebar_size_in_units=scalebar_size_in_units, 
                                    scalebar_thickness=.01)
    
    limits = [[np.percentile(a, 1), np.percentile(a, 99)] for a in after]
    im2 = np.stack(after)
    microim2 = microplot.Microimage(images=im2, 
                                    proj_type='sum', 
                                    cmaps=cmap, 
                                    rescale_type='limits',
                                    limits=limits, 
                                    label_text='', 
                                    label_color='white', 
                                    channel_label_show=True, 
                                    unit='um', 
                                    scalebar_unit_per_pix=scalebar_um_per_pix, 
                                    scalebar_size_in_units=scalebar_size_in_units, 
                                    scalebar_thickness=.01)

    panel = Micropanel(rows=1, cols=2, figsize=[10, 10], figscaling=3)
    panel.add_element([0,0], microim1);
    panel.add_element([0,1], microim2);

    



    

