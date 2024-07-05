%% MERIT Data Preprocessing
%% Author: Ranyu FAN
%% Date: 2023


clear
clc

%% PATH

% run('../pathFinder.m');

% zone = 'test_area_01_removedZ'; % subject_to_change
% areaWTD = 'NAMERICA'; % SUBJECT_TO_CHANGE

%MERITfolder         = '/Data/thnguyen/data/process_FI/';
MERITFolder          ='/home/gsainton/MERIT'; 
% WTDfolder           = '/Data/thnguyen/WaterTableDepth/';

L8Folder            = '/home/gsainton/CALER/PEATMAP/0_data_preprocessing/1_GEE_DP/';
fileR               = [L8Folder '/LS8_raw/LS8_B1_5.0.tif'];

processedDataFolder = 'MERIT_welldone';

%% Permanent water

filePermWater = [processedDataFolder '/permWater.mat'];

% Read the fileR which is a .tif file 
[A, R] = readgeoraster(fileR);
% Plot the fileR
%figure
%mapshow(A, R);

%disp(R);
% Get the latitude and longitude limits of the fileR
%minLat = min(R.LatitudeLimits);

%load(fileR, 'R');
minLat = min(R.LatitudeLimits);
maxLat = max(R.LatitudeLimits);
minLon = min(R.LongitudeLimits);
maxLon = max(R.LongitudeLimits);
res    = R.CellExtentInLatitude;

disp(minLat);
disp(maxLat);
disp(minLon);
disp(maxLon);
disp(res);


%Original data
[permWater, lat, lon] = windowCell(MERITFolder, minLat, maxLat, minLon, maxLon, 'wth');
permWater(permWater==-1 | permWater>0) = 1;
[lon, lat] = meshgrid(lon, lat);

%New grid
newLat   = linspace(maxLat - res/2, minLat + res/2, R.RasterSize(1));
newLon   = linspace(minLon + res/2, maxLon - res/2, R.RasterSize(2));
[newLon, newLat] = meshgrid(newLon, newLat);
permWater = round(interp2(lon, lat, permWater, newLon, newLat));
newLat   = linspace(maxLat - res/2, minLat + res/2, R.RasterSize(1));
newLon   = linspace(minLon + res/2, maxLon - res/2, R.RasterSize(2));

save(filePermWater, 'permWater', 'R', '-v7.3');
fprintf('SAVED %s\n', filePermWater);

clear R

%% Area Correction: exclude permanent water

fprintf('Correct: recompute peat fraction by excluding permanent water')

peatFolder = '../2_PEATMAP_DP';

filePeatmapPoly2  = [peatFolder '/PEATMAP_welldone/' zone '_latlon.shp'];

fileRasterGeoref  = [peatFolder '/PEATMAP_welldone/' zone '_georef.tif'];
fileRasterFrac    = [peatFolder '/PEATMAP_welldone/' zone '_frac.tif'];

fileRasterFracC   = [peatFolder '/PEATMAP_welldone/' zone '_frac_corrected.tif'];
filePeatmapPolyC  = [peatFolder '/PEATMAP_welldone/' zone '_latlon_corrected.shp'];

[fracRaster, R] = readgeoraster(fileRasterFrac);
georefRaster = readgeoraster(fileRasterGeoref);
peatmap = shaperead(filePeatmapPoly2); 
fracRaster = single(fracRaster);

for i = 1:length(peatmap)
    fprintf('Polygon %d/%d\n', i, length(peatmap));
    ind = find(georefRaster==i);
    permBox = permWater(ind);
    permPer = sum(permBox==1)/length(ind)*100;
    if permPer==100; fprintf('All permanent water inside this polygon!\n'); end
    newFrac = min(peatmap(i).PEAT_PER/(100-permPer)*100, 100);
    fracRaster(ind) = newFrac;
    peatmap(i).PEAT_PER = newFrac;
end

geotiffwrite(fileRasterFracC, fracRaster, R);
fprintf('SAVED %s\n', fileRasterFracC);
shapewrite(peatmap, filePeatmapPolyC); 

clear permWater fracRaster georefRaster peatmap R

%% DISTANCE TO SMALL DRAINAGE

myFile = [processedDataFolder '/dist0005.mat'];

load(fileR, 'R');
minLat = min(R.LatitudeLimits);
maxLat = max(R.LatitudeLimits);
minLon = min(R.LongitudeLimits);
maxLon = max(R.LongitudeLimits);
res    = R.CellExtentInLatitude;

%Original data
[dist0005, lat, lon] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, 'dest_0005');
[lon, lat] = meshgrid(lon, lat);

%New grid
newLat   = linspace(maxLat - res/2, minLat + res/2, R.RasterSize(1));
newLon   = linspace(minLon + res/2, maxLon - res/2, R.RasterSize(2));
[newLon, newLat] = meshgrid(newLon, newLat);

dist0005 = interp2(lon, lat, dist0005, newLon, newLat);
save(myFile, 'dist0005', 'R', '-v7.3');
fprintf('SAVED %s\n', myFile);

clear dist0005 R

%% DISTANCE TO MEDIUM DRAINAGE

myFile = [processedDataFolder '/dist0100.mat'];

load(fileR, 'R');
minLat = min(R.LatitudeLimits);
maxLat = max(R.LatitudeLimits);
minLon = min(R.LongitudeLimits);
maxLon = max(R.LongitudeLimits);
res    = R.CellExtentInLatitude;

%Original data
[dist0100, lat, lon] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, 'dest_0100');
[lon, lat] = meshgrid(lon, lat);

%New grid
newLat   = linspace(maxLat - res/2, minLat + res/2, R.RasterSize(1));
newLon   = linspace(minLon + res/2, maxLon - res/2, R.RasterSize(2));
[newLon, newLat] = meshgrid(newLon, newLat);

dist0100 = interp2(lon, lat, dist0100, newLon, newLat);
save(myFile, 'dist0100', 'R', '-v7.3');
fprintf('SAVED %s\n', myFile);

clear dist0100 R 

%% DISTANCE TO LARGE DRAINAGE

myFile = [processedDataFolder '/dist1000.mat'];

load(fileR, 'R');
minLat = min(R.LatitudeLimits);
maxLat = max(R.LatitudeLimits);
minLon = min(R.LongitudeLimits);
maxLon = max(R.LongitudeLimits);
res    = R.CellExtentInLatitude;

%Original data
[dist1000, lat, lon] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, 'dest_1000');
[lon, lat] = meshgrid(lon, lat);

%New grid
newLat   = linspace(maxLat - res/2, minLat + res/2, R.RasterSize(1));
newLon   = linspace(minLon + res/2, maxLon - res/2, R.RasterSize(2));
[newLon, newLat] = meshgrid(newLon, newLat);

dist1000 = interp2(lon, lat, dist1000, newLon, newLat);
save(myFile, 'dist1000', 'R', '-v7.3');
fprintf('SAVED %s\n', myFile);

clear dist1000

%% HEIGHT ABOVE SMALL DRAINAGE

myFile = [processedDataFolder '/hand0005.mat'];

load(fileR, 'R');
minLat = min(R.LatitudeLimits);
maxLat = max(R.LatitudeLimits);
minLon = min(R.LongitudeLimits);
maxLon = max(R.LongitudeLimits);
res    = R.CellExtentInLatitude;

%Original data
[hand0005, lat, lon] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, 'hand_0005');
[lon, lat] = meshgrid(lon, lat);

%New grid
newLat   = linspace(maxLat - res/2, minLat + res/2, R.RasterSize(1));
newLon   = linspace(minLon + res/2, maxLon - res/2, R.RasterSize(2));
[newLon, newLat] = meshgrid(newLon, newLat);

hand0005 = interp2(lon, lat, hand0005, newLon, newLat);
save(myFile, 'hand0005', 'R', '-v7.3');
fprintf('SAVED %s\n', myFile);

clear hand0005 R myFile

%% HEIGHT ABOVE MEDIUM DRAINAGE

myFile = [processedDataFolder '/hand0100.mat'];

load(fileR, 'R');
minLat = min(R.LatitudeLimits);
maxLat = max(R.LatitudeLimits);
minLon = min(R.LongitudeLimits);
maxLon = max(R.LongitudeLimits);
res    = R.CellExtentInLatitude;

%Original data
[hand0100, lat, lon] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, 'hand_0100');
[lon, lat] = meshgrid(lon, lat);

%New grid
newLat   = linspace(maxLat - res/2, minLat + res/2, R.RasterSize(1));
newLon   = linspace(minLon + res/2, maxLon - res/2, R.RasterSize(2));
[newLon, newLat] = meshgrid(newLon, newLat);

hand0100 = interp2(lon, lat, hand0100, newLon, newLat);
save(myFile, 'hand0100', 'R', '-v7.3');
fprintf('SAVED %s\n', myFile);

clear hand0100 R myFile

%% HEIGHT ABOVE LARGE DRAINAGE

myFile = [processedDataFolder '/hand1000.mat'];

load(fileR, 'R');
minLat = min(R.LatitudeLimits);
maxLat = max(R.LatitudeLimits);
minLon = min(R.LongitudeLimits);
maxLon = max(R.LongitudeLimits);
res    = R.CellExtentInLatitude;

%Original data
[hand1000, lat, lon] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, 'hand_1000');
[lon, lat] = meshgrid(lon, lat);

%New grid
newLat   = linspace(maxLat - res/2, minLat + res/2, R.RasterSize(1));
newLon   = linspace(minLon + res/2, maxLon - res/2, R.RasterSize(2));
[newLon, newLat] = meshgrid(newLon, newLat);

hand1000 = interp2(lon, lat, hand1000, newLon, newLat);
save(myFile, 'hand1000', 'R', '-v7.3');
fprintf('SAVED %s\n', myFile);

clear hand1000 R myFile

%% SLOPE

myFile = [processedDataFolder '/slope.mat'];

load(fileR, 'R');
minLat = min(R.LatitudeLimits);
maxLat = max(R.LatitudeLimits);
minLon = min(R.LongitudeLimits);
maxLon = max(R.LongitudeLimits);
res    = R.CellExtentInLatitude;

%Original data
[slope, lat, lon] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, 'slope');
[lon, lat] = meshgrid(lon, lat);

%New grid
newLat   = linspace(maxLat - res/2, minLat + res/2, R.RasterSize(1));
newLon   = linspace(minLon + res/2, maxLon - res/2, R.RasterSize(2));
[newLon, newLat] = meshgrid(newLon, newLat);

slope = interp2(lon, lat, slope, newLon, newLat);
save(myFile, 'slope', 'R', '-v7.3');
fprintf('SAVED %s\n', myFile);

clear slope

%% ELEVATION

myFile = [processedDataFolder '/elevation.mat'];

load(fileR, 'R');
minLat = min(R.LatitudeLimits);
maxLat = max(R.LatitudeLimits);
minLon = min(R.LongitudeLimits);
maxLon = max(R.LongitudeLimits);
res    = R.CellExtentInLatitude;

%Original data
[elevation, lat, lon] = windowCell(MERITfolder, minLat, maxLat, minLon, maxLon, 'elevation');
[lon, lat] = meshgrid(lon, lat);

%New grid
newLat   = linspace(maxLat - res/2, minLat + res/2, R.RasterSize(1));
newLon   = linspace(minLon + res/2, maxLon - res/2, R.RasterSize(2));
[newLon, newLat] = meshgrid(newLon, newLat);

elevation = interp2(lon, lat, elevation, newLon, newLat);
save(myFile, 'elevation', 'R', '-v7.3');
fprintf('SAVED %s\n', myFile);

clear elevation
