
import os, sys
import rasterio
import numpy as np

class HydroCell:
    """
    Class to read a cell of MERIT Hydro
    INPUTS:
    ----------
    :folder_path: path to the folder containing the data
    :cell_lat: latitude of the cell
    :cell_lon: longitude of the cell
    :ext: extension of the data
    OUTPUTS:
    ----------
    :data: data of the cell
    :nrows: number of rows of the cell
    :ncols: number of columns of the cell
    :nbits: number of bits of the data
    :nodata: no data value
    :xdim: x dimension of the cell
    :ydim: y dimension of the cell
    :ulx_map: x coordinate of the upper left corner of the cell
    :ulymap: y coordinate of the upper left corner of the cell
    :x: x coordinates of the cell
    :y: y coordinates of the cell
    """


    def __init__(self, folder_path, cell_lat, cell_lon, ext):
        """
        Initialize the class
        INPUTS:
        ----------
        :folder_path: path to the folder containing the data
        :cell_lat: latitude of the cell
        :cell_lon: longitude of the cell
        :ext: extension of the data
        
        """
        self.folder_path = folder_path
        self.cell_lat = cell_lat
        self.cell_lon = cell_lon
        self.ext = ext

        if self.cell_lat > 90:
            self.cell_lat = self.cell_lat - 90
        if self.cell_lon >= 180:
            self.cell_lon = self.cell_lon - 360
        if self.cell_lon < -180:
            self.cell_lon = self.cell_lon + 360

        lat_letter = 'n' if self.cell_lat >= 0 else 's'
        lon_letter = 'e' if self.cell_lon >= 0 else 'w'

        self.cell_name = f"{lat_letter}{abs(int(self.cell_lat)):02d}{lon_letter}{abs(int(self.cell_lon)):03d}"
        self.file_type = ext

        if self.file_type[:3] == 'dow':
            self.nbits = 8
            self.folder_path = '/Data/thnguyen/data/LEO/'

        self.ulx_map = self.cell_lon
        self.uly_map = self.cell_lat + 5 - 3 / 3600  # Defaulting to the grid of MERIT

        self.dtype = 'float'

        if ext == 'slope':
            self.layout = 'TIF'
            ftif = f"{self.folder_path}Yamazaki/MERIT/v0.4_diagnostics_old_tif/slope_tif/{self.cell_name}.tif"
        elif ext == 'wth':
            self.layout = 'TIF'
            ftif = f"{self.folder_path}Yamazaki/MERIT/v0.4_original_data_distributed/width/{self.cell_name}_wth.tif"
        elif ext == 'flwdir':
            self.layout = 'TIF'
            ftif = f"{self.folder_path}Yamazaki/MERIT/v0.4_original_data_distributed/flwdir/{self.cell_name}_dir.tif"
        elif ext == 'uparea':
            self.layout = 'TIF'
            ftif = f"{self.folder_path}Yamazaki/MERIT/v0.4_original_data_distributed/uparea/{self.cell_name}_upa.tif"
        elif ext == 'elevation':
            self.layout = 'TIF'
            ftif = f"{self.folder_path}Yamazaki/MERIT/v0.4_original_data_distributed/elevtn/{self.cell_name}_elv.tif"
        elif ext[:3] in ['des', 'han']:
            self.layout = 'TIF'
            ftif = f"{self.folder_path}Yamazaki2/{ext}/{self.cell_name}.tif"
        elif ext[:3] == 'for':
            self.layout = 'BIL'
            fbil = f"{self.folder_path}{ext}/{self.cell_name}.bil"
            fhdr = f"{self.folder_path}{ext}/{self.cell_name}.hdr"
        else:
            self.layout = 'BIL'
            fbil = f"{self.folder_path}{ext.upper()}_DATA/{self.cell_name}_{ext}.bil"
            fhdr = f"{self.folder_path}{ext.upper()}_DATA/{self.cell_name}_{ext}.hdr"

        if self.layout == 'BIL':
            zip_hdr = 0
            zip_bil = 0

            if os.path.exists(fhdr + '.gz'):
                os.system(f"gunzip -fk {fhdr}.gz")
                zip_hdr = 1

            if os.path.exists(fbil + '.gz'):
                os.system(f"gunzip -fk {fbil}.gz")
                zip_bil = 1

            elif os.path.exists(fbil + '.tar.gz'):
                extract_folder = f"{self.folder_path}{ext}_DATA/"
                os.system(f"tar -xzvf {fbil}.tar.gz --strip 5 -C {extract_folder}")
                zip_bil = 1

            if os.path.exists(fbil) and os.path.exists(fhdr):
                with open(fhdr, 'r') as hdr_file:
                    hdr = hdr_file.readlines()

                    for line in hdr:
                        field = line.split()[0]

                        if field == 'nrows':
                            self.nrows = int(line.split()[1])
                        elif field == 'ncols':
                            self.ncols = int(line.split()[1])
                        elif field == 'nbits':
                            self.nbits = int(line.split()[1])
                        elif field == 'nodata':
                            self.nodata = float(line.split()[1])
                        elif field == 'ulxmap':
                            self.ulx_map = float(line.split()[1])
                        elif field == 'ulymap':
                            self.ulx_map = float(line.split()[1])
                        elif field == 'xdim':
                            self.xdim = float(line.split()[1])
                        elif field == 'ydim':
                            self.ydim = float(line.split()[1])

                if self.nbits == 0:
                    self.dtype = 'float'
                else:
                    self.dtype = f'int{self.nbits}'

                with open(fbil, 'rb') as bil_file:
                    self.data = np.fromfile(bil_file, dtype=self.dtype).reshape((self.nrows, self.ncols))

                self.data[self.data == self.nodata] = np.nan

            else:
                if not os.path.exists(fbil):
                    print(f'File "{fbil}" does not exist.')
                if not os.path.exists(fhdr):
                    print(f'File "{fhdr}" does not exist.')

                self.data = np.zeros((self.nrows, self.ncols))

            if zip_hdr == 1:
                os.system(f"rm {fhdr}")

            if zip_bil == 1:
                os.system(f"rm {fbil}")

        elif self.layout == 'TIF':
            if os.path.exists(ftif):
                with rasterio.open(ftif) as tif_file:
                    self.data = tif_file.read(1)
                    self.nrows, self.ncols = self.data.shape
                    self.nbits = tif_file.dtypes[0].itemsize * 8
                    self.xdim, self.ydim = tif_file.transform[0], -tif_file.transform[4]
                    self.ulx_map, self.ulx_map = tif_file.bounds.left + self.xdim / 2, tif_file.bounds.top - self.ydim / 2

                if ext == 'flwdir':
                    self.dtype = 'uint8'
                    self.nodata = 247
                else:
                    self.dtype = f'float{self.nbits}'
                    self.nodata = -9999
                    self.data[self.data == self.nodata] = np.nan

            else:
                print(f'File "{ftif}" does not exist.')

        self.x = np.arange(self.ulx_map, self.ulx_map + self.xdim * self.ncols, self.xdim)
        self.y = np.arange(self.ulymap, self.ulymap - self.ydim * self.nrows, -self.ydim)


def window_cell(path, lat_min0, lat_max0, lon_min0, lon_max0, ext):
    """
    Extract a window of data from a cell
    INPUTS:
    ----------
    :path: path to the folder containing the data
    :lat_min0: minimum latitude of the window
    :lat_max0: maximum latitude of the window
    :lon_min0: minimum longitude of the window
    :lon_max0: maximum longitude of the window
    :ext: extension of the data
    OUTPUTS:
    ----------
    :windo: window of data
    :lat_values: latitude values of the window
    :lon_values: longitude values of the window

    """
    dsharp = 30 / 3600 if ext == 'glwd' else (15 / 3600 if ext == 'giemsD15' else 3 / 3600)
    cell_size = 5 / dsharp

    lat_min = np.floor(lat_min0 / dsharp + 0.5) * dsharp
    lat_max = np.floor(lat_max0 / dsharp + 0.5) * dsharp
    lon_min = np.floor(lon_min0 / dsharp + 0.5) * dsharp
    lon_max = np.floor(lon_max0 / dsharp + 0.5) * dsharp

    n_rows = int(np.floor((lat_max - lat_min) / dsharp + 0.5))
    n_cols = int(np.floor((lon_max - lon_min) / dsharp + 0.5))
    windo = np.zeros((n_rows, n_cols), dtype=np.float32)

    lat_box_max = int(np.ceil(lat_max / 5) * 5)
    lat_box_min = int(np.floor(lat_min / 5) * 5)
    lon_box_max = int(np.ceil(lon_max / 5) * 5)
    lon_box_min = int(np.floor(lon_min / 5) * 5)

    for lat in range(lat_box_min, lat_box_max, 5):
        for lon in range(lon_box_min, lon_box_max, 5):
            cell = HydroCell(path, lat, lon, ext) 
            padding_left = int(np.floor((lon_min - lon) / dsharp + 0.5))
            padding_right = int(np.floor((lon_max - lon) / dsharp + 0.5))
            padding_top = int(-np.floor((lat_max - lat - 5) / dsharp + 0.5))
            padding_bottom = int(-np.floor((lat_min - lat - 5) / dsharp + 0.5))

            windo[max(0, 0 - padding_top):min(n_rows, int(np.floor((lat_max - lat) / dsharp + 0.5))),
                  max(0, 0 - padding_left):min(n_cols, int(np.floor((lon + 5 - lon_min) / dsharp + 0.5)))] \
                = cell.data[max(0, 0 + padding_top):min(cell_size, padding_bottom),
                            max(0, padding_left + 1):min(cell_size, padding_right)]

    lat_values = np.arange(lat_max + dsharp / 2, lat_max - n_rows * dsharp / 2, -dsharp)
    lon_values = np.arange(lon_min - dsharp / 2, lon_min + n_cols * dsharp / 2, dsharp)

    return windo, lat_values.astype(np.float32), lon_values.astype(np.float32)


def freadl(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()
    

