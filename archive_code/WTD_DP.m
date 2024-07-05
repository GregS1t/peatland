%% WTD Data Preprocessing

clear
clc

%% PATH

run('../pathFinder.m');



% zone = 'test_area_01_removedZ'; % subject_to_change
% areaWTD = 'NAMERICA'; % SUBJECT_TO_CHANGE

% MERITfolder         = '/Data/thnguyen/data/process_FI/';
% WTDfolder           = '/Data/thnguyen/WaterTableDepth/';

L8Folder            = '../1_GEE_DP/LS8_welldone';
fileR               = [L8Folder '/landsat' prefix '_m5.mat'];

processedDataFolder = 'WTD_welldone';

%% WATER TABLE DEPTH

myFile = [processedDataFolder '/wtd.mat'];

load(fileR, 'R');
minLat = min(R.LatitudeLimits);
maxLat = max(R.LatitudeLimits);
minLon = min(R.LongitudeLimits);
maxLon = max(R.LongitudeLimits);
res    = R.CellExtentInLatitude;

%Original data
[wtd, lat, lon, mask] = waterTableDepth(WTDfolder, minLat, maxLat, minLon, maxLon, areaWTD, 0);
[lon, lat] = meshgrid(lon, lat);

%New grid
newLat   = linspace(maxLat - res/2, minLat + res/2, R.RasterSize(1));
newLon   = linspace(minLon + res/2, maxLon - res/2, R.RasterSize(2));
[newLon, newLat] = meshgrid(newLon, newLat);

wtd  = interp2(lon, lat, wtd, newLon, newLat);
mask = interp2(lon, lat, single(mask), newLon, newLat); mask = int8(round(mask));
wtd(~mask) = NaN;
save(myFile, 'wtd', 'R', '-v7.3');
fprintf('SAVED %s\n', myFile)

%clear wtd R lat lon mask
