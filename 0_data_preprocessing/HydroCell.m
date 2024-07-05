classdef HydroCell
%Cellule 5x5 d'Hydrosheds

properties
    x;
    y;
    data;
    byteorder='I';
    layout; %'TIF' for MERIT data and 'BIL' for all others
    nrows=6000;
    ncols=6000;
    nbits=32;
    dtype;
    nodata=-9999;
    ulxmap; %longitude at CENTER of upper left pixel
    ulymap; %latitude at CENTER of upper left pixel
    xdim=3/3600; %convert 3arcsecond to degree
    ydim=3/3600;
    cellLat; %de -90 a 90
    cellLon; %de -180 a 180
    folderPath; %'/Data/thnguyen/data/process_FI/', dossier de donnees
    cellName; %n00e035 ...
    fileType; %DEM, ACC ...
end



methods
    function obj=HydroCell(folderPath, cellLat,cellLon,ext)

        obj.folderPath = folderPath;

        obj.cellLat=cellLat;
        if (cellLat>90)
           obj.cellLat=cellLat-90;
        end
        obj.cellLon=cellLon;
        if (cellLon>=180)
            obj.cellLon=cellLon-360;
        elseif (cellLon<-180)
            obj.cellLon=cellLon+360;
        end

        latLetter='n';
        if (obj.cellLat<0)
            latLetter='s';
        end
        lonLetter='e';
        if (obj.cellLon<0)
            lonLetter='w';
        end

        obj.cellName = strcat(latLetter,sprintf('%02d',abs(obj.cellLat)),lonLetter,sprintf('%03d',abs(obj.cellLon)));
        obj.fileType = ext;
	 
        if strcmp(obj.fileType(1:3),'dow')
            obj.nbits=8;
            obj.folderPath = '/Data/thnguyen/data/LEO/';
        end

        %obj.ulxmap=obj.cellLon  +obj.xdim/2;  %center of upper left pixel
        %obj.ulymap=obj.cellLat+5-obj.ydim/2;
        obj.ulxmap=obj.cellLon;                %center of upper left pixel
        obj.ulymap=obj.cellLat+5-obj.ydim;  %use grid of MERIT as default. 
                                            %Lower & leftward 1/2 pixel compare to HydroSHED's grid

        obj.dtype = 'float';
        
        %-------------------------------------------------------------------------------------------------------------
	    %File path
        if strcmp(ext, 'slope')
	        obj.layout = 'TIF';
		    ftif = strcat(obj.folderPath, ...
		                  'Yamazaki/MERIT/v0.4_diagnostics_old_tif/slope_tif/',   obj.cellName, '.tif');
	    elseif strcmp(ext, 'wth')
	        obj.layout = 'TIF';
		    ftif = strcat(obj.folderPath, ...
		                  'Yamazaki/MERIT/v0.4_original_data_distributed/width/', obj.cellName, '_wth.tif');
		elseif strcmp(ext, 'flwdir') %uint8
		    obj.layout = 'TIF';
		    ftif = strcat(obj.folderPath, ...
		                  'Yamazaki/MERIT/v0.4_original_data_distributed/flwdir/', obj.cellName, '_dir.tif');
		elseif strcmp(ext, 'uparea')
		    obj.layout = 'TIF';
		    ftif = strcat(obj.folderPath, ...
		                  'Yamazaki/MERIT/v0.4_original_data_distributed/uparea/', obj.cellName, '_upa.tif');
		elseif strcmp(ext, 'elevation')
		    obj.layout = 'TIF';
		    ftif = strcat(obj.folderPath, ...
		                  'Yamazaki/MERIT/v0.4_original_data_distributed/elevtn/', obj.cellName, '_elv.tif');
	    elseif (strcmp(ext(1:3), 'des') || strcmp(ext(1:3), 'han'))
	        obj.layout = 'TIF';
		    ftif = strcat(obj.folderPath, 'Yamazaki2/', ext, '/', obj.cellName, '.tif');
	    elseif strcmp(ext(1:3), 'for')
	        obj.layout = 'BIL';
            fbil = strcat(obj.folderPath, ext, '/', obj.cellName, '.bil');
            fhdr = strcat(obj.folderPath, ext, '/', obj.cellName, '.hdr');        	        
        else
            obj.layout = 'BIL';
            fbil = strcat(obj.folderPath, upper(ext), '_DATA/', obj.cellName,'_',ext,'.bil');
            fhdr = strcat(obj.folderPath, upper(ext), '_DATA/', obj.cellName,'_',ext,'.hdr');        
	    end


        %Check and Read file
        %-------------------Data from Leo -----------------------------------------------------------------   
        if strcmp(obj.layout, 'BIL')
            %Desarchiver le fichier
            zip_hdr=0; zip_bil=0;
            if exist([fhdr '.gz'],'file')
                command=['gunzip -fk ',fhdr,'.gz']; system(command);
                zip_hdr=1;
            end
            if exist([fbil '.gz'],'file')
                command=['gunzip -fk ',fbil,'.gz']; system(command);
                zip_bil=1;
            elseif exist([fbil '.tar.gz'],'file')
                extract_folder = strcat(obj.folderPath, upper(ext), '_DATA/');
                command=['tar -xzvf ',   fbil, '.tar.gz --strip 5 -C ' extract_folder]; system(command);
                zip_bil=1;
            end
            
            if (exist(fbil,'file') && exist(fhdr,'file'))
                %Read hdr
                hdr = freadl(fhdr);
                %on parcours les differents champs du header
                for i = 1:length(hdr)
                    field = sscanf(hdr{i},'%s %*s');
                    switch lower(field)
                        case 'nrows',  obj.nrows  = sscanf(hdr{i},'%*s%u');
                        case 'ncols',  obj.ncols  = sscanf(hdr{i},'%*s%u');
                        case 'nbits',  obj.nbits  = sscanf(hdr{i},'%*s%u');
                        case 'nodata', obj.nodata = sscanf(hdr{i},'%*s%f');
                        case 'ulxmap', obj.ulxmap = sscanf(hdr{i},'%*s%f');
                        case 'ulymap', obj.ulymap = sscanf(hdr{i},'%*s%f');
                        case 'xdim',   obj.xdim   = sscanf(hdr{i},'%*s%f');
                        case 'ydim',   obj.ydim   = sscanf(hdr{i},'%*s%f');
                    end
                end
               % Options
                if (obj.nbits==0) 
                    obj.dtype='float';
                else
                    obj.dtype = sprintf('int%u',obj.nbits); %exemple, int32
                end
                % Read data
                fid = fopen(fbil,'rb','n'); %fichier bil
                obj.data = fread(fid,[obj.ncols,obj.nrows],obj.dtype)';        %for output always in double #freadDouble
                %obj.data = fread(fid,[obj.ncols,obj.nrows],['*' obj.dtype])'; %for output in the same format as obj.dtype
                fclose(fid);
                %if strcmp(obj.dtype, 'float')   %disable (i.e. comment) this line if #freadDouble
                    obj.data(obj.data==obj.nodata) = NaN;
                %end
            else
                if ~exist(fbil,'file'), fprintf('File "%s" does not exist. \n',fbil); end
                if ~exist(fhdr,'file'), fprintf('File "%s" does not exist. \n',fhdr); end
                %nouveau fichier
                %obj.data=zeros(obj.nrows,obj.ncols);
                obj.data=nan(obj.nrows,obj.ncols); %FIXME new
            end
            if zip_hdr==1; command=['rm ', fhdr]; system(command); end
            if zip_bil==1; command=['rm ', fbil]; system(command); end
        %-------------------Data from Yamazaki ---------------------------------------------------------------   
        elseif strcmp(obj.layout, 'TIF')
            if exist(ftif, 'file')
                [obj.data,~] = geotiffread(ftif);
                tifInfo      = geotiffinfo(ftif);
                obj.nrows    = tifInfo.Height;
                obj.ncols    = tifInfo.Width;
                obj.nbits    = tifInfo.BitDepth;
                obj.xdim     = tifInfo.PixelScale(1);
                obj.ydim     = tifInfo.PixelScale(2);
                obj.ulxmap   = tifInfo.BoundingBox(1,1)+obj.xdim/2;
                obj.ulymap   = tifInfo.BoundingBox(2,2)-obj.ydim/2; 
                if strcmp(ext, 'flwdir')
                    obj.dtype  = 'uint8';
                    obj.nodata = 247;
                else                
                    obj.dtype  = sprintf('float%u',obj.nbits); %Yamazaki used float in all (dist, hand, slope)
                    obj.nodata = -9999;
                    obj.data(obj.data==obj.nodata) = NaN;
                end
            else
                fprintf('File "%s" does not exist. \n',ftif);
                %nouveau fichier
                %obj.data=zeros(obj.nrows,obj.ncols);
                obj.data=nan(obj.nrows,obj.ncols); %FIXME new
            end
        end         
        % Create x,y vectors  (coordinates at center of pixel)
        obj.x = obj.ulxmap+obj.xdim*(0:obj.ncols-1);   
        obj.y = obj.ulymap-obj.ydim*(0:obj.nrows-1)';
    end





    function []= writeCellBil(obj, wfolderPath)
        if wfolderPath(end) == '/'; wfolderPath(end)=[]; end
        wfolderPath = [wfolderPath '/' upper(obj.fileType), '_DATA/'];
        create_dir(wfolderPath);
        fileHeader=fopen([wfolderPath,'/',obj.cellName,'_',obj.fileType,'.hdr'],'w');
        fileBil   =fopen([wfolderPath,'/',obj.cellName,'_',obj.fileType,'.bil'],'w');

        %Ecriture du BIL
        if contains(obj.dtype, 'float')
            obj.dtype = 'float'; %no matter float32 or float64, always save float (i.e float32 single)
            obj.nbits = 0; 
        end
            obj.data(isnan(obj.data)) = obj.nodata;  %put this line inside the 'if' above if NOT #freadDouble
        fwrite(fileBil, obj.data, obj.dtype);
        fclose(fileBil);

        if (strcmp(obj.fileType(1:3),'dow'))
            % archiver le bil
            command=['tar czfP ', fileBil, '.tar.gz ', fileBil];
            system(command);
            % Supprimer le bil
            %command=['rm -f ',pathBil];
            %system(command);
        else
            gzip([wfolderPath,'/',obj.cellName,'_',obj.fileType,'.bil']);
            system(['rm ' [wfolderPath,'/',obj.cellName,'_',obj.fileType,'.bil']]);
        end

        %Ecriture du HDR
        fprintf(fileHeader,'BYTEORDER        %s\n',obj.byteorder);
        fprintf(fileHeader,'LAYOUT        %s\n', 'BIL');
        fprintf(fileHeader,'NROWS        %u\n',obj.nrows);
        fprintf(fileHeader,'NCOLS        %u\n',obj.ncols);
        fprintf(fileHeader,'NBITS       %u\n',obj.nbits);
        fprintf(fileHeader,'NODATA        %f\n',obj.nodata);
        fprintf(fileHeader,'ULXMAP        %f\n',obj.ulxmap);
        fprintf(fileHeader,'ULYMAP        %f\n',obj.ulymap);
        fprintf(fileHeader,'XDIM        %f\n',obj.xdim);
        fprintf(fileHeader,'YDIM        %f\n',obj.ydim);
        fclose(fileHeader);
        gzip([wfolderPath,'/',obj.cellName,'_',obj.fileType,'.hdr']);
        system(['rm ' [wfolderPath,'/',obj.cellName,'_',obj.fileType,'.hdr']]);
    end




    function []= writeCellTif(obj, wfolderPath)
        create_dir(wfolderPath);
        pathTif   =strcat(wfolderPath,'/', obj.cellName, '.tif');

        if contains(obj.dtype, 'float')
            obj.data(isnan(obj.data)) = obj.nodata;
        else
            eval(['obj.data = ' obj.dtype '(obj.data);']); %do this if #freadDouble
        end
                
        R = georasterref('RasterSize', [obj.nrows,obj.ncols], ...
                         'RasterInterpretation', 'cells', 'ColumnsStartFrom', 'north', ...
                         'LatitudeLimits', [(obj.y(end)-obj.ydim/2), (obj.y(1)  +obj.ydim/2)], ...
                         'LongitudeLimits',[(obj.x(1)  -obj.xdim/2), (obj.x(end)+obj.xdim/2)]);

        geotiffwrite(pathTif, obj.data, R);
        fprintf('SAVED %s \n', pathTif)
    end





    function []=quickPlot(obj)
        if strcmp(obj.layout, 'BIL')
            cellDir= HydroCell(obj.cellLat, obj.cellLon, 'dir');
            cellDir.data(isnan(cellDir.data))=cellDir.nodata;
        else
            cellDir=HydroCell(obj.cellLat,obj.cellLon,'flwdir');
        end
        % Si c'est une cellule de GLWD
        if strcmp(obj.fileType,'glwd')
            data0=kron(obj.data,ones(10,10)); %insert 'obj.dtype' in ones() if NOT #freadDouble
            dll=(5/6000)*(1:6000);
            h=imagesc(obj.ulxmap-obj.xdim/2+dll,obj.ulymap+obj.ydim/2-dll,data0);% plot from center of 1st px to of last px
            set(gca,'YDir','normal');
            if (sum(sum(cellDir.data==cellDir.nodata))>0)
                set(h,'AlphaData',~(cellDir.data==cellDir.nodata));
            end
            axis('equal');
            axis('image');
            xlabel('Longitude (\circ)','FontSize',14);
            ylabel('Latitude (\circ)','FontSize',14);

            wetlands=unique(obj.data);
            nbTicks=length(wetlands);
            glwdLegend={'Dry','Lake','Reservoir','River','Freshwater Marsh, Floodplain', ...
                        'Swamp Forest, Flooded Forest','Coastal Wetland','Pan, Brackish/Saline Wetland', ...
                        'Bog, Fen, Mire (Peatland)','Intermittent Wetland/Lake', ...
                        '50-100% Wetland','25-50% Wetland','0-25% Wetland'};
            lbls=glwdLegend(wetlands+1);
            cmap=jet(nbTicks);
            if min(wetlands)==0
                cmap(1,:)=[1,1,1];
            end
            colormap(cmap);
            L = line(obj.cellLon*ones(nbTicks),obj.cellLat*ones(nbTicks), 'LineWidth',14);
            set(L,{'color'}, mat2cell(cmap(wetlands+1,:), ones(1, length(wetlands)),3));
            legend(lbls,'Location','eastoutside','FontSize',14);
        % Si c'est une cellule de flow accumulation d'HydroSHEDS 15arcsec
        elseif strcmp(obj.fileType,'acc')
            data0=kron(obj.data,ones(5,5)); %insert 'obj.dtype' in ones() if NOT #freadDouble
            dll=(5/6000)*(1:6000);
            h=imagesc(obj.ulxmap-obj.xdim/2+dll,obj.ulymap+obj.ydim/2-dll,data0);% plot from center of 1st px to of last px
            set(gca,'YDir','normal');
            if (sum(sum(cellDir.data==cellDir.nodata))>0)
                set(h,'AlphaData',~(cellDir.data==cellDir.nodata));
            end
            axis('equal');
            axis('image');
            xlabel('Longitude (\circ)','FontSize',14);
            ylabel('Latitude (\circ)','FontSize',14);
            clb=colorbar;
            ylabel(clb,'[pixels]','FontSize',14);       
        else
            % Les valeurs de distance a vol d'oiseau aux riviers sont exprimees en decametre: convertion en km
            data=obj.data;
            if strcmp(obj.fileType(1:3),'dest')
                data=data/100;
            end
            %h=imagesc(obj.ulxmap+obj.xdim*(1:obj.ncols),obj.ulymap-obj.ydim*(1:obj.nrows),data);
            h=imagesc(obj.x, obj.y, data);
            if (sum(sum(cellDir.data==cellDir.nodata))>0)
                set(h,'AlphaData',~(cellDir.data==cellDir.nodata));
            end
            axis('equal');
            axis('image');
            xlabel('Longitude (\circ)','FontSize',14);
            ylabel('Latitude (\circ)','FontSize',14);
            set(gca,'YDir','normal');
            % Affichage de la colorbar
            if strcmp(obj.fileType(1:3),'dow')
                cmap=jet(2);
                cmap(1,:)=[1,1,1];
                cmap(2,:)=[0,0,0];
                colormap(cmap);
                L = line(obj.cellLon*ones(2),obj.cellLat*ones(2), 'LineWidth',12);
                set(L,{'color'},mat2cell(cmap,ones(1,2),3));
                legend({'Dry','Inu.'},'Location','northeast');
            else
                clb=colorbar;
                % Label de la colorbar
                switch obj.fileType(1:3)
                    case 'des'
                        ylabel(clb,'Distance [km]','FontSize',14);
                    case 'han'
                        ylabel(clb,'Elevation [m]','FontSize',14);
                    case 'slo'
                        ylabel(clb,'Slope [m]','FontSize',14);
                    case 'flw'
                        ylabel(clb,'Direction','FontSize',14);
                end
            end %if dow
        end %if glwd
    end %function quickplot
    
end% method

end%class
