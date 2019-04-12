import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def visualize_img_activations(img, activations, cmap=plt.cm.jet, 
                              luminance_scale=(None, None),
                              fontsize=24, act_title="Activations", title=""):
    
    if len(img.shape) == 1:
        h, w = img.shape
        c = None
    elif len(img.shape) == 3:
        c, h, w = img.shape
    else:
        raise NotImplementedError
    
    img = img.transpose(0,1).transpose(1,2)
    
    if c is None or c == 1:
        cmap = "gray"
        img = img[:,:,0]
        
    k, ah, aw = activations.shape
    vmin, vmax = luminance_scale
    
    # Activations subplot: nrow x ncol
    nrows = int(np.ceil(np.sqrt(k)))
    ncols = int(np.ceil(1. * k / nrows))
    
    # Input image span
    img_rows = nrows if nrows < 16 else nrows // 2
    img_cols = img_rows
    
    # Main grid dimensions
    grid_rows, grid_cols = nrows, ncols + img_cols
    gridspec.GridSpec(grid_rows, grid_cols)
    
    # Plot image
    plt.subplot2grid((grid_rows, grid_cols), (0,0), colspan=img_cols, rowspan=img_rows)
    
    if vmin:
        plt.imshow(img, cmap, interpolation=None, vmin=vmin, vmax=vmax)
    else:
        plt.imshow(img, cmap, interpolation=None)
        
    plt.xticks([]), plt.yticks([])
    plt.ylabel("Input ", fontsize=int(fontsize * 0.75))
    
    # Plot activations
    for r in range(nrows):
        for c in range(ncols):
            
            a_idx = r * ncols + c
            if a_idx < len(activations):
                plt.subplot2grid((grid_rows, grid_cols), (r, img_cols+c), colspan=1, rowspan=1)
                if r == 0 and c == int(np.floor(ncols/2))-1:
                    plt.title(act_title, fontsize=int(fontsize * 0.75))
                if vmin:
                    plt.imshow(activations[a_idx], cmap, interpolation=None, vmin=vmin, vmax=vmax)
                else:
                    plt.imshow(activations[a_idx], cmap, interpolation=None)
                    
            plt.xticks([]), plt.yticks([])
    plt.suptitle(title, fontsize=fontsize)
