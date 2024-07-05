#!/usr/bin/env python
# Name: LANDSAT_Extract.py - Version 1.0
# Author: Grégory Sainton
# Mail : gregory.sainton@obspm.fr   
# Date: 2023-11-13
# Description: Extract the data from the LANDSAT8 images from Google Earth Engine

# Version: 1.0

import os, sys  # the operating system library
import ee   # the Earth Engine Python API library
import numpy as np # the numpy library
import seaborn as sns # the seaborn library
import geetools # the geetools library

import gdown # the gdown library to download data from Google Drive
#import gdal # the gdal library to read raster data
from osgeo import gdal
def maskL8sr(image):
    """
    Function to mask clouds using the pixel_qa band of Landsat 8 SR data.
    https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_SR

    Parameters
    ----------
    image: Image
        The Landsat 8 SR image

    Returns
    -------
    Image
        The cloudmasked Landsat 8 image
    """

    # Mask the input for clouds, cloud shadows, snow
    fillBitMask         = (1 << 0) #0000000000000001       BIT 0: FILL
    dilatedCloudBitMask = (1 << 1) #0000000000000010       BIT 1: DILATED CLOUD  
    unusedBitMask       = (1 << 2) #0000000000000100       BIT 2: UNUSED
    cloudsBitMask       = (1 << 3) #0000000000001000       BIT 3: CLOUD
    cloudShadowBitMask  = (1 << 4) #0000000000010000       BIT 4: CLOUD SHADOW 
    snowBitMask         = (1 << 5) #0000000000100000       

    # Get the pixel QA band.
    qa = image.select('QA_PIXEL')
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0) \
        .And(qa.bitwiseAnd(cloudsBitMask).eq(0)) \
        .And(qa.bitwiseAnd(snowBitMask).eq(0)) \
        .And(qa.bitwiseAnd(fillBitMask).eq(0))

    # Saturation masks QA_RADSAT band
    # Get the RADSAT band
    radsat = image.select('QA_RADSAT')
    # Apply mask
    mask = mask.And(radsat.eq(0)) # 0 = not saturated

    # Generate the optical and thermal bands to masked image and return it.
    # Scale factor for reflectance conversion to get optical values
    # https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2#bands

    # Estimate the optical bands designed
    opticalBands = image.select('SR_B.') \
        .multiply(0.0000275).add(-0.2)
    
    # Estimate the thermal band
    thermalBand = image.select('ST_B10') \
        .multiply(0.00341802).add(149.0) \
        
    # Add those bands to image
    image = image.addBands(opticalBands, None, True)\
        .addBands(thermalBand, None, True)

    return image.updateMask(mask)

# Define the function to add NDVI band
def addNDVI(image):
    ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# Group the images by month
def groupByMonth(imageCollection):
    def func(image, newlist):
        date = ee.Date(image.get('system:time_start'))
        month = date.get('month')
        return ee.List(newlist).add(ee.Image(image.set('month', month)))
    return ee.List(imageCollection.iterate(func, ee.List([])))


def extract_ls8_data(AOI, start_date, end_date, 
                     start_month, end_month, scale=90,
                     folder='ls8_data'):
    """
    Function to extract the data from the LANDSAT8 images from Google Earth Engine

    Parameters
    ----------
    AOI: list - The Area Of Interest (AOI)
    start_date: str - The start date 
    end_date: str - The end date
    start_month: int - The start month for each year
    end_month: int - The end month for each year
    scale: int - The scale of the images
    folder: str - OUtput folder name in Google Drive

    Returns
    -------
    The data are exported to Google Drive

    """

    # Define the Landsat 8 Surface Reflectance collection
    # https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_SR
    ls8_collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(AOI) \
        .filterDate(start_date, end_date) \
        .map(maskL8sr) \
        .select('SR_B.') \
        .map(lambda image: image.clip(AOI)) \
        
    # Get the number of images
    count = ls8_collection.size().getInfo()
    print("Number of images: ", count)

    months = ee.List.sequence(start_month, end_month) 
    print("Months: ", months.size().getInfo())

    # Group the images by month
    # Make a list of images for each month
    byMonth = ee.ImageCollection.fromImages(
        months.map(lambda m: ls8_collection.filter(ee.Filter.calendarRange(m, m, 'month')).median()\
               .multiply(10000).toInt16().unmask(-32768).clip(AOI)\
               .set('month', m))
    )

    # Create a list of band numbers
    listBand = ee.List.sequence(1, 7)
    myRegion = AOI.bounds().getInfo()

    # Get the information about the AOI bounds
    aoi_bounds = AOI.bounds()

    # Convert bounds to GeoJSON format
    aoi_geojson = aoi_bounds.getInfo()['coordinates']
    print("Start exporting to Google Drive.")
    # Iterate over the list of bands
    from tqdm import tqdm
    for i in tqdm(range(listBand.length().getInfo())):
        # Get the current band number
        myBand = 'SR_B' + str(listBand.get(i).getInfo())
        print(myBand)
        myImCol = byMonth.select(myBand)

        # Get the number of images
        count = myImCol.size().getInfo()
        #print("Number of images in the collection: ", count)

        # Export the image collection to Google Drive
        _ = geetools.batch.Export.imagecollection.toDrive(
                                        myImCol, 
                                        folder=folder, 
                                        namePattern='LS8_B' + str(listBand.get(i).getInfo()) + "_{month}", 
                                        region=aoi_geojson, 
                                        scale=scale, # resolution in meters/pixel
                                        maxPixels=int(1e13))
        
    # Monitor the task
    print("Exporting to Google Drive. Please check the Tasks tab in \
          the Earth Engine Code Editor for progress.")

def get_data_from_google_drive(folder_id):
    """
    Function to get the data from Google Drive

    Parameters
    ----------
    None

    Returns
    -------
    The data are exported to Google Drive

    """
   
    # Download the data from Google Drive
    url = f'https://drive.google.com/folders/{folder_id}'
    output = os.path.join(os.getcwd(), 'LS8_raw')
    gdown.download(url, output, quiet=False, use_cookies=False)
    

def landsat_aggregate_per_month(input_folder, output_folder):
    """
    Function to preprocess the Landsat 8 data

    This function agregate the image of the different bands into a single
    file for a given month.

    This function is not really flexible and should be adapted to the specific 
    names of the files, the range of months and the possible extensions due 
    to the size of the images from Google Earth Engine.


    Parameters
    ----------
    input_folder: str - The input folder
    output_folder: str - The output folder

    Returns
    -------
    The data are save in the output folder
    
    """
    from scipy.io import savemat

    # MUST BE DEFINE BY THE USER
    file_middle_names = [".0"]
    range_of_months = [5, 9]
    PREFIX = "LS8_B"
    extension_file = ".tif"
    
    
    number_of_names = len(file_middle_names)
    

    number_of_months = range_of_months[1] - range_of_months[0] + 1

    for m in range(range_of_months[0], range_of_months[1] + 1):
        file_l8 = os.path.join(output_folder, f'landsat_m{m}.mat')

        for iband in range(1, 8):
            A = []
            for k in range(number_of_names):
                file_path = os.path.join(input_folder, f'{PREFIX}{iband}_{m}{file_middle_names[k]}{extension_file}')
                dataset = gdal.Open(file_path)
                A.append((dataset.GetRasterBand(1).ReadAsArray(), dataset.GetGeoTransform()))
                dataset = None

            for l in range(number_of_names - 1):
                A[l + 1] = (np.hstack((A[l][0], A[l + 1][0])), A[l][1])

            B = A[number_of_names - 1][0]

            if m == range_of_months[0] and iband == 1:
                latlim = A[0][1][3], A[number_of_names - 1][1][3] + A[number_of_names - 1][1][5] * A[number_of_names - 1][0].shape[0]
                lonlim = A[0][1][0], A[number_of_names - 1][1][0] + A[number_of_names - 1][1][1] * A[number_of_names - 1][0].shape[1]
                raster_size = A[0][0].shape[0], A[0][0].shape[1] * (number_of_names - 1) + A[number_of_names - 1][0].shape[1]
                R = (latlim[0], (latlim[1] - latlim[0]) / raster_size[0], 0.0, lonlim[0], 0.0, (lonlim[1] - lonlim[0]) / raster_size[1])

                allL8 = np.zeros((raster_size[0] * raster_size[1], 7), dtype=np.int16)

            allL8[:, iband - 1] = B.flatten()

        info = {'CornerCoords': {'Lat': latlim, 'Lon': lonlim},
                'PixelScale': [R[1], R[5]],
                'Height': raster_size[0],
                'Width': raster_size[1]}

        #np.savez(file_l8, allL8=allL8, R=R, info=info)
        savemat(file_l8, {'allL8': allL8, 'R': R, 'info': info})
        print(f'SAVED {file_l8}')


if __name__ == "__main__":
    print("Attention, je suis une librairie, pas un programme !")