# MIT Licence (https://mit-license.org/)
# ----------------------------------------------------------
#
# Author: Grégory Sainton
# Mail : gregory.sainton[at]obspm.fr
# Date: 2023-11-13
# Description: Library of functions for the PEATLAND_PROJECT
#
# Version: 1.0

import os, sys  # the operating system library
from pprint import pprint 


import ee   # the Earth Engine Python API library
import numpy as np # the numpy library
import matplotlib.pyplot as plt # the matplotlib library
import pandas as pd # the pandas library

import geopandas as gpd
from shapely.geometry import box

import rasterio
from rasterio.features import rasterize

def shapesfile2df(PEATMAP_DATA_DIR, verbose=False, export=False,
                  export_path=None, export_name=None):
    """
    Function to extract the bounding box of each shapefile
    and store it in a dataframe
    
    INPUT
    ----------
    PEATMAP_DATA_DIR: path to the directory containing the shapefiles of the peatlands
    verbose: boolean - print the content of the shapefile
    export: boolean - export the dataframe to a xml file
    export_path: path to the directory where to export the xml file
    export_name: name of the xml file to export
    
    OUTPUT
    ----------
    df_shapefiles: dataframe containing the bounding box of each shapefile,
    the file name and the path

    """

    # List of shapefiles over all directories
    list_shapefiles = []
    for root, dirs, files in os.walk(PEATMAP_DATA_DIR):
        for file in files:
            if file.endswith('.shp'):
                list_shapefiles.append(os.path.join(root, file))
    if verbose:
        print("Number of shapefiles: ", len(list_shapefiles))

        # print the list of files found
        pprint(list_shapefiles)

    # Open the shapefile and print the content the bounding box
    df_shapefiles = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", 
                                          "zone_name", "country_name", 
                                          "file_name", "path"])
    for file in list_shapefiles:
        geozone = gpd.read_file(file)
        if verbose:
            print("Shapefile: ", file)
            pprint(geozone.crs)
        geozone = geozone.to_crs("EPSG:4326")   
        if verbose:
            print(geozone.total_bounds)
            print("")

        # Get Zone name from the path, the zone name is the last directory name

        # Split the path
        path_split = file.split("/")
        if verbose:
            print("Path split: ", path_split)
        # Get the index of the last directory
        index = len(path_split) - 2
        # Get the last directory name
        zone_name = path_split[index]
        if verbose:
            print("Zone name: ", zone_name)

        # Get the country name from the path, the country name is the last directory name
        # Get the index of the last directory
        index = len(path_split) - 1
        # Get the last directory name and remove the extention
        country_name = path_split[index].split(".")[0]
        # Remove _Peatland from the name
        country_name = country_name.replace("_Peatland", "")

        if verbose:
            print("Country name: ", country_name)


        # Get the file name
        file_name = os.path.basename(file)
        if verbose:
            print("File name: ", file_name)

        # Get the bounding box  
        xmin, ymin, xmax, ymax = geozone.total_bounds
        if verbose:
            print("xmin: ", xmin)
            print("ymin: ", ymin)
            print("xmax: ", xmax)
            print("ymax: ", ymax)
            print("")
        
        entry = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, 
                 "zone_name": zone_name, "country_name": country_name, 
                 "file_name": file_name, "path": file}
        df_shapefiles = pd.concat([df_shapefiles, pd.DataFrame(entry, index=[0])], ignore_index=True)
    if verbose:
        print(df_shapefiles)


    if export:
        df_shapefiles.to_xml(os.path.join(export_path, export_name), index=False)  

    return df_shapefiles



def plot_all_shapefiles(df_shapefiles):
    """
    Function to plot all the bounding box from the dataframe on a world map

    INPUT
    ----------
    df_shapefiles: dataframe containing the bounding box of each shapefile,
    the file name and the path

    OUTPUT
    ----------
    world_map: world map with all the bounding box plotted
    """
    
    import folium # the folium library

    # Plot all the bounding box from the dataframe on a world map
    # Create a world map using folium
    world_map = folium.Map(zoom_start=3)

    # Loop over the dataframe to add the bounding box to the map with different colors

    # Define a colormap
    colormap = ["red", "blue", "green", "orange", "purple", "darkred", 
                "darkblue", "darkgreen", "cadetblue", "pink",
                "lightred", "lightblue", "lightgreen", "gray",
                "black", "lightgray"]


    for index, row in df_shapefiles.iterrows():
        # Get the bounding box
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        # Get the country name
        country_name = row["country_name"]
        # Get the zone name
        zone_name = row["zone_name"]
        # Get the file name
        file_name = row["file_name"]

        # Define AOI with the coordinates of the bounding box
        AOI = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
        # Overlap the AOI on the world map with a different color for each shapefile
        folium.GeoJson(AOI.getInfo(), name=zone_name, 
                    style_function=lambda x: {'color':colormap[index]}).add_to(world_map)


        #folium.GeoJson(AOI.getInfo(), name=zone_name).add_to(world_map)
        # Add a label to the AOI
        folium.Marker(location=[(ymin + ymax)/2, (xmin + xmax)/2], 
                    popup=f"{country_name}/{zone_name}").add_to(world_map)
        

    # Display the map
    display(world_map)  


def unzip_files(dir_path):
    """AOI: unzip the files and rename the directories

    INPUT:
    ------
    dir_path: str - path to the directory containing the zip files

    OUTPUT:
    -------
    list_zip_files : list - list of shapefiles to process 
    
    """

    list_zip_files = [f for f in os.listdir(dir_path) if f.endswith('.zip')]

    if len(list_zip_files) != 0:
        for file in list_zip_files:
            print("Unzip file: ", file)
            os.system("unzip " + os.path.join(dir_path, file) + " -d " + dir_path)

        # Create a directory to store the zip files
        os.system("mkdir " + os.path.join(dir_path, "zip_files"))

        # replace the spaces by underscore in directory names
        for root, dirs, files in os.walk(dir_path):
            for dir in dirs:
                if " " in dir:
                    print("Rename directory: ", dir)
                    os.rename(os.path.join(root, dir), os.path.join(root, dir.replace(" ", "_")))


        # move the zip files to the zip_files directory
        for file in list_zip_files:
            os.system("mv " + os.path.join(dir_path, file) + " " + os.path.join(dir_path, "zip_files"))

        return list_zip_files   

    else:
        print("No zip files to unzip")
        return None


def check_zone_coverage(coordinates, df_shapefiles):
    """
    Function to check if a zone defined by its coordinates is covered by one of the shapefiles
    
    Parameters:
    ----------
    coordinates (dict): Dictionary containing the coordinates of the zone (xmin, ymin, xmax, ymax)
    df_shapefiles (pd.DataFrame): DataFrame containing the shapefile information
    
    Returns:
    ----------
    tuple: Tuple containing:
        covered (bool): True if the zone is covered by one of the shapefiles, False otherwise
        dict_zone (dict): Dictionary containing the coordinates of the zone covered by the shapefile,
                            the name of the zone, the name of the shapefile and the surface of the zone

    """
    covered = False
    dict_zone = {}
    min_area = 1e10
    for index, row in df_shapefiles.iterrows():
        xmin = row["xmin"]
        ymin = row["ymin"]
        xmax = row["xmax"]
        ymax = row["ymax"]
        if xmin <= coordinates["xmin"] <= xmax \
            and ymin <= coordinates["ymin"] <= ymax \
                and xmin <= coordinates["xmax"] <= xmax \
                    and ymin <= coordinates["ymax"] <= ymax:
            covered = True
            zone_name = row["zone_name"]
            file_name = row["file_name"]
            file_path = row["path"]
            area = (xmax - xmin) * (ymax - ymin)
            if area < min_area:
                min_area = area
                dict_zone = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, 
                         "zone_name": zone_name, "file_name": file_name, "file_path": file_path,
                         "surface":min_area}
            
    return covered, dict_zone



def crop_shapefile(geozone, aoi, plot=False, 
                   export=False, export_path=None, 
                   export_name=None):
    """
    Function to crop a shapefile with respect to the AOI
    
    Parameters:
    ----------
    geozone : geopandas.GeoDataFrame - shapefile to crop
    aoi (dict): Dictionary containing the coordinates of 
    the AOI (xmin, ymin, xmax, ymax)
    
    Returns:
    ----------
    cropped_shapefile (geopandas.GeoDataFrame): Cropped shapefile
    
    """
    # Create a bounding box from the AOI
    aoi_bbox = box(aoi["xmin"], aoi["ymin"], aoi["xmax"], aoi["ymax"])
    aoi_bbox = gpd.GeoDataFrame(geometry=[aoi_bbox], crs="EPSG:4326")
    # Crop the shapefile with the AOI
    cropped_shapefile = gpd.overlay(geozone, aoi_bbox, how='intersection')

    if plot:

        # Plot the shapefile with projection WGS 84
        fig, ax = plt.subplots(figsize=(10, 10))
        geozone.plot(ax=ax, color='green')
        ax.set_title("Map of PEATLANDS in Canada")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(geozone.total_bounds[0], geozone.total_bounds[2])
        ax.set_ylim(geozone.total_bounds[1], geozone.total_bounds[3])
        ax.grid()

        # Create a geodataframe from the bounding box

        aoi_bbox.plot(ax=ax, color='red', edgecolor='black', alpha=0.5)
        # Subplot with the cropped shapefile
        axins = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
        cropped_shapefile.plot(ax=axins, color='red',
                    edgecolor='black', alpha=0.5)
        axins.set_xlim(aoi["xmin"], aoi["xmax"])
        axins.set_ylim(aoi["ymin"], aoi["ymax"])
        axins.set_xticklabels('')
        axins.set_yticklabels('')
        axins.grid()
        ax.indicate_inset_zoom(axins)

        plt.tight_layout()
        plt.show()

    if export:
        if not os.path.exists(export_path):
            os.makedirs(export_path)

        if os.path.exists(os.path.join(export_path, export_name)):
            overwrite = input("File already exists, overwrite? (y/n)")
            if overwrite in ["y", "Y", "yes", "Yes", "YES"]:
                cropped_shapefile.to_file(os.path.join(export_path, export_name))
            else:
                print("File not overwritten")
        else:
            cropped_shapefile.to_file(os.path.join(export_path, export_name))

    
    return cropped_shapefile

def control_rasterized(shp2raster, src):
    """ 
    Function to control the rasterized shapefile
    
    Parameters:
    ----------
    shp2raster (str): Path to the rasterized shapefile
    src (rasterio.DatasetReader): Rasterio object
    
    Returns:
    ----------
    None
    """
    if shp2raster.res == src.res:
        print("Resolution OK")
    else:
        print("Resolution not OK")
        print("Resolution expected: ", src.res)
        print("Resolution found: ", shp2raster.res)

    if shp2raster.crs == src.crs:
        print("CRS OK")
    else:
        print("CRS not OK")
        print("CRS expected: ", src.crs)
        print("CRS found: ", shp2raster.crs)

    if shp2raster.bounds == src.bounds:
        print("Bounds OK")
    else:
        print("Bounds not OK")
        print("Bounds expected: ", src.bounds)
        print("Bounds found: ", shp2raster.bounds)

    if shp2raster.shape == src.shape:
        print("Shape OK")
    else:
        print("Shape not OK")
        print("Shape expected: ", src.shape)
        print("Shape found: ", shp2raster.shape)


def rasterize_shapefile(shapefile, src, output_raster_path, pixel_size,
                        attribute=None, verbose=False, control=False):
    """ 
    Function to rasterize a shapefile
    
    Parameters:
    ----------
    shapefile : geopandas.GeoDataFrame - shapefile to rasterize
    output_raster_path (str): Path to save the output raster
    pixel_size (float): Resolution of the output raster
    attribute (str): Attribute column to use for raster values
    
    Returns:
    ----------
    None
    """
    # Open the shapefile
    geozone = shapefile
    # Get the bounds of the shapefile
    #bounds = geozone.total_bounds
    bounds = src.bounds

    profile = {
        'driver': 'GTiff',
        'count': 1,
        'dtype': rasterio.uint8,
        'nodata': 0,
        'crs': geozone.crs,
        'width': src.shape[1],
        'height': src.shape[0],
        'transform': rasterio.transform.from_origin(bounds[0], bounds[3], 
                                                    pixel_size, pixel_size)
    }

    # Create an empty raster array
    raster_data = np.zeros((profile['height'], profile['width']), dtype=np.uint8)

    # If an attribute column is specified, use it to fill the raster values
    if attribute is not None:
        # Get the attribute values from the shapefile
        if attribute != "ALL":
            # check if attribute in the shapefile
            if attribute not in geozone.columns:
                print("Attribute {} not in the shapefile".format(attribute))
                print("List of attributes: ", geozone.columns)
                sys.exit()

            if verbose:
                print("Attribute: {}".format(attribute))
                
            values = geozone[attribute].values
            # Rasterize the shapefile with the attribute values
            shapes = [(geom, value) for geom, value in zip(geozone['geometry'],
                                                            values)]
            raster_data = rasterize(
                shapes=shapes,
                out_shape=(profile['height'], profile['width']),
                transform=profile['transform'],
                fill=0,
                all_touched=True,
                default_value=0
            )
            # Create a raster file and write the data
                            # Create a raster file and write the data
            output = os.path.splitext(output_raster_path)[0]
            output = output + "_" + attribute + ".tif"
            output = output.replace(" ", "_")  
            with rasterio.open(output_raster_path, 'w', **profile) as dst:
                dst.write(raster_data, 1)
                if verbose:
                    print("Raster data saved in ", output_raster_path)
            if control:
                control_rasterized(rasterio.open(output_raster_path), src)

        else:
           
            list_columns = geozone.columns.tolist()
            list_columns.remove("geometry")
            if verbose:
                print(list_columns)

            for column in list_columns:
                values = geozone[column].values
                # Rasterize the shapefile with the attribute values
                shapes = [(geom, value) for geom, value in zip(geozone['geometry'],
                                                                values)]
                raster_data = rasterize(
                    shapes=shapes,
                    out_shape=(profile['height'], profile['width']),
                    transform=profile['transform'],
                    fill=0,
                    all_touched=True,
                    default_value=0
                )
                # Create a raster file and write the data
                output = os.path.splitext(output_raster_path)[0]
                output = output + "_" + column + ".tif"
                output = output.replace(" ", "_")  
                
                with rasterio.open(output, 'w', **profile) as dst:
                    dst.write(raster_data, 1)
                    if verbose:
                        print("Raster data saved in ", output)
                if control:
                    control_rasterized(rasterio.open(output), src)
    else:
        print("Please specify an attribute column")
        sys.exit()

