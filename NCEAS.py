import numpy as np
from pyproj import Transformer
from pathlib import Path
import rasterio



def coord_trans(x, y, order="CH_to_normal"):

    if order == "CH_to_normal":
        transformer = Transformer.from_crs("EPSG:21781", "EPSG:4326", always_xy=True)
        x_out, y_out = transformer.transform(x, y)  # X=Easting, Y=Northing
    elif order == "normal_to_CH":
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:21781", always_xy=True)
        x_out, y_out = transformer.transform(x, y)  # X=Longitude, Y=Latitude
    else:
        raise ValueError("order must be either 'CH_to_normal' or 'normal_to_CH'")
    return x_out, y_out

def coord_trans_shift_old(x,y, order="CH_to_normal"):
    "converts from the NCEAS dataset coordinates (they have a shift) to regular lat, lon (and vice versa)"
    shift_x, shift_y= (1011627.4909483634, -100326.1477937577) #See "coordinates.ipynb"
    if order=="CH_to_normal":
        lons, lats = coord_trans(x-shift_x, y-shift_y,order="CH_to_normal")
        return lons, lats
    elif order =="normal_to_CH":
        x_trans, y_trans=coord_trans(x, y, order= "normal_to_CH")
        return x_trans+shift_x, y_trans+shift_y
    
def coord_trans_shift(x,y, order="CH_to_normal"):
    "converts from the NCEAS dataset coordinates (they have a shift) to regular lat, lon (and vice versa)"
    shift_x, shift_y= (1010927.4909483634, -101026.1477937577) # +700 for each, kinda random
    if order=="CH_to_normal":
        lons, lats = coord_trans(x-shift_x, y-shift_y,order="CH_to_normal")
        return lons, lats
    elif order =="normal_to_CH":
        x_trans, y_trans=coord_trans(x, y, order= "normal_to_CH")
        return x_trans+shift_x, y_trans+shift_y
    

    
def NCEAS_covariates(lon, lat, directory="embeddings_data_and_dictionaries/data_SDM_NCEAS/Environnement", return_dict=True):
    """
    Fetch raster values for given lon/lat coordinates from all .tif files in a directory.
    
    Parameters:
        lon (array-like): Longitudes
        lat (array-like): Latitudes
        directory (str or Path): Path to folder containing .tif raster files
    
    Returns: either a dataframe or an np array, depending on the argumennt "return_dict"
        dict: Keys are raster filenames (without extension), values are arrays of sampled values
        
    """
    directory = Path(directory)
    
    # Transform coordinates
    x, y = coord_trans_shift(lon, lat, order="normal_to_CH") #error?
    points = np.column_stack([x, y])  # shape (n_points, 2)
    
    # Get all .tif files
    tif_files = sorted(directory.glob("*.tif"))
    
    if len(tif_files) == 0:
        print("No TIFF files found in directory:", directory)
        return {}
    
    # Dictionary to hold values from all rasters
    all_values = {}
    
    for tif_path in tif_files:
        with rasterio.open(tif_path) as src:
            # Sample raster at given points
            values = np.array([v[0] for v in src.sample(points)])
            all_values[tif_path.stem] = values  # use filename without extension as key
    
    if return_dict:
        return all_values
    else:
        # Stack values into a 2D array: shape (num_points, num_rasters)
        # Order of columns matches tif_files
        array_values = np.column_stack([all_values[tif_path.stem] for tif_path in tif_files])
        return array_values