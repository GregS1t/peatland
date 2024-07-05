function [wtd, lat, lon, mask] = waterTableDepth(WTD_folder, minLat, maxLat, minLon, maxLon, myZone, plottt)

%Water table depth of each month in 2004 (monthlymeans) or averaged over the year 2004 (annual mean)

% addpath('/obs/thnguyen/these/4_Draft/');
% WTD_folder = '/Data/thnguyen/WaterTableDepth/';

%i=2;
%WTD_file = [WTD_folder char(myZone(i)) '_WTD_monthlymeans.nc'];
WTD_file = [WTD_folder char(myZone) '_WTD_annualmean.nc'];

ncdisp(WTD_file);
%info = ncinfo(WTD_file);
%scale_factor = info.Variables(end).Attributes(3).Value;
%add_offset   = info.Variables(end).Attributes(4).Value;

lon = ncread(WTD_file, 'lon');
lat = ncread(WTD_file, 'lat'); %from South to North 
[~, indLatStart] = min(abs(lat-minLat)); 
[~, indLatStop]  = min(abs(lat-maxLat));
[~, indLonStart] = min(abs(lon-minLon)); 
[~, indLonStop]  = min(abs(lon-maxLon));

lat = single(lat(indLatStart:indLatStop)); lat = flipud(lat);
lon = single(lon(indLonStart:indLonStop));

mask = ncread(WTD_file, 'mask', [indLonStart indLatStart], [length(lon) length(lat)]);
mask = rot90(mask);

wtd = ncread(WTD_file, 'WTD', [indLonStart indLatStart 1], [length(lon) length(lat) Inf]);
wtd = single(rot90(wtd)*-1);%*scale_factor + add_offset;
%for iMonth = 1:12
%    toto = wtd(:,:, iMonth);
%    toto(~mask) = NaN;
%    wtd(:,:,iMonth) = toto;
%end
wtd(~mask) = NaN;

if plottt ==1
    myColormap = hex2rgb({'#004FB3', '#00a7ff', '#7FBB00', '#748F2E', '#FCE49F', '#FCAF3E', '#F57900', '#EF2929'});
    toto = zeros(size(mask));
    toto(wtd(:,:,1)<=0.25)=1;
    toto(wtd(:,:,1)>0.25 & wtd(:,:,1)<=2.5)=2;
    toto(wtd(:,:,1)>2.5  & wtd(:,:,1)<=5)  =3;
    toto(wtd(:,:,1)>5    & wtd(:,:,1)<=10) =4;
    toto(wtd(:,:,1)>10   & wtd(:,:,1)<=20) =5;
    toto(wtd(:,:,1)>20   & wtd(:,:,1)<=40) =6;
    toto(wtd(:,:,1)>40   & wtd(:,:,1)<=80) =7;
    toto(wtd(:,:,1)>80)=8;
    figure; set(gcf, 'Color', 'w', 'Units', 'normalized',  'OuterPosition', [0 0 1 1]);
    h = imagesc(toto); set(h, 'AlphaData', mask==1); colormap(myColormap); clim([1 8]); colorbar;
end

end %function

%fileTiff = [WTD_folder '/WTD_finland.tif'];
%res = lat(2)-lat(1);
%R = georasterref('RasterSize', size(wtd), ...
%                 'RasterInterpretation', 'cells', 'ColumnsStartFrom', 'north', ...
%                 'LatitudeLimits',  double([min(lat(:))-res/2 max(lat(:))+res/2]), ...
%                 'LongitudeLimits', double([min(lon(:))-res/2 max(lon(:))+res/2]));
%geotiffwrite(fileTiff, wtd, R);
