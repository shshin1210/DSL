%============================================================================
%
% Codename: Bird Scanner (Yale Computer Graphics Group)
%
% Copyright (C) 2011-12 Min H. Kim. All rights reserved.
%
% GNU General Public License Usage
% Alternatively, this file may be used under the terms of the GNU General
% Public License version 3.0 as published by the Free Software Foundation
% and appearing in the file LICENSE.GPL included in the packaging of this
% file. Please review the following information to ensure the GNU General
% Public License version 3.0 requirements will be met:
% http://www.gnu.org/copyleft/gpl.html.
%
%============================================================================
% function merge_segments(type, sourcepath, pieceheight, pieceoverlap, rect, totalres, tttime, nopad)
function merge_segments(type, sourcepath, pieceheight, pieceoverlap, rect, totalres, tttime, nopad)
    regionheight = rect(4)+1;

    % ask filename for the reconstructed listfile
    if nargin<=1;
        % file read
        [file_name,file_path] = uigetfile({'*_rec.txt','txt';},'Choose reconstructed filelist');
        % leave function when no file is selected.
        if size(file_path,2) == 1
            return;
        end
        sourcepath = char(strcat(file_path,file_name));   
    end
    
    % read the list text file
    %recfilelist = textread('filelist_rec.txt', '%s'); % NB this is cell data
    recfilelist = textread(sourcepath, '%s'); % NB this is cell data

    disp 'done';


    % load one file for paramters
    load (recfilelist{1});
    
    % total size
    inum = size(recfilelist, 1);
    iint = size(x_twist,1);
    
    
   %{
    %disp('calculating means...');
    tx_twist = zeros(total, size(x_twist,2), size(x_twist,3),'single');
    medenergy = zeros(inum,size(tx_twist,3));
    % find mean intensity in each piece
    if (inum~=1)
        for i=1:(inum)
            disp(num2str(i));
            load (recfilelist{i});
            % remove negative values
            x_twist = x_twist.*(x_twist>=0);
            if i==1
                stcom = 1;
                edcom = a;
                stsam = 1;
                edsam = a - b;
                disp(['stsam: ' num2str(stsam)]);
                disp(['edsam: ' num2str(edsam)]);
                disp(['stpie: ' num2str(1)]);
                disp(['edpie: ' num2str((a-b))]);
                temp = x_twist(1:(a-b),:,:);
                temp = reshape(temp, size(temp,1)*size(temp,2),size(temp,3));
                medenergy(i,:) = mean(temp);
    %            disp(['medenergy: ' num2str(medenergy(i,:))]);
            elseif i==inum
                stcom = (i-1)*a - (i-1)*2*b + 1;
                edcom = regionheight;
                stsam = (i-1)*a - ((i-1)*2-1)*b + 1;
                edsam = stsam + size(x_twist,1) - b - 1; %regionheight;
                disp(['stsam: ' num2str(stsam)]);
                disp(['edsam: ' num2str(edsam)]);
                disp(['stpie: ' num2str(1+b)]);
                disp(['edpie: ' num2str(a-b)]);
                %temp = x_twist(1+b:(edsam-stsam+b+1),:,:);
                temp = x_twist(1+b:size(x_twist,1),:,:);
                temp = reshape(temp, size(temp,1)*size(temp,2),size(temp,3));
                medenergy(i,:) = mean(temp);
    %            disp(['medenergy: ' num2str(medenergy(i,:))]);
            else            
                stcom = (i-1)*a - (i-1)*2*b + 1;
                edcom = i*a - (i-1)*2*b;
                stsam = (i-1)*a - ((i-1)*2-1)*b + 1;
                edsam = i*a - ((i-1)*2+1)*b;
                disp(['stsam: ' num2str(stsam)]);
                disp(['edsam: ' num2str(edsam)]);
                disp(['stpie: ' num2str(1+b)]);
                disp(['edpie: ' num2str(a-b)]);
                temp = x_twist(1+b:(a-b),:,:);
                temp = reshape(temp, size(temp,1)*size(temp,2),size(temp,3));
                medenergy(i,:) = mean(temp);
     %           disp(['medenergy: ' num2str(medenergy(i,:))]);
            end            
        end
    end
    % this should be individual channel scalar
    totalmed = mean(medenergy);
    
    % check this out
    scalarmat = abs(medenergy./repmat(totalmed,[size(medenergy,1) 1])); % SCALARS
    scalarmat = ones(size(scalarmat));% killing scalars --> WE DON'T NEED INDIVIDUAL SCALARS
%}        
    tttime = 0;
  
    if (inum==1)
        load (recfilelist{1});
        tttime = tttime + ttime;
        tx_twist = x_twist;
    else

        %disp('calculating means...');
        for i=1:(inum)
            disp(num2str(i));
            load (recfilelist{i});
            tttime = tttime + ttime;
            if i==1
                stcom = 1;
                edcom = pieceheight;
                stsam = 1;
                edsam = pieceheight - pieceoverlap;
                disp(['stsam: ' num2str(stsam)]);
                disp(['edsam: ' num2str(edsam)]);
                disp(['stsam: ' num2str(1)]);
                disp(['edsam: ' num2str((pieceheight - pieceoverlap))]);

                % apply the scalar on individual spectral channels
%                 scalar = scalarmat(i,:);
%                 h = size(x_twist,1);
%                 w = size(x_twist,2);
%                 d = size(x_twist,3);
%                 x_twist = reshape(x_twist,[h*w, d]);
%                 scalar = repmat(scalar,[h*w 1]);
%                 x_twist = scalar.*x_twist;
%                 x_twist = reshape(x_twist,[h,w,d]);

                tx_twist(stsam:edsam, :, :) = x_twist(1:(pieceheight - pieceoverlap),:,:);
            elseif i==inum && inum~=1
                stcom = (i-1)*pieceheight - (i-1)*2*pieceoverlap + 1;
                edcom = regionheight;
                stsam = (i-1)*pieceheight - ((i-1)*2-1)*pieceoverlap + 1;
                edsam = stsam + size(x_twist,1) - pieceoverlap - 1; %regionheight;
                disp(['stsam: ' num2str(stsam)]);
                disp(['edsam: ' num2str(edsam)]);
                disp(['stsam: ' num2str(1+pieceoverlap)]);
                disp(['edsam: ' num2str(pieceheight - pieceoverlap)]);

                % apply the scalar on individual spectral channels
%                 scalar = scalarmat(i,:);
%                 h = size(x_twist,1);
%                 w = size(x_twist,2);
%                 d = size(x_twist,3);
%                 x_twist = reshape(x_twist,[h*w, d]);
%                 scalar = repmat(scalar,[h*w 1]);
%                 x_twist = scalar.*x_twist;
%                 x_twist = reshape(x_twist,[h,w,d]);

                tx_twist(stsam:edsam, :, :) = x_twist(1+pieceoverlap:size(x_twist,1),:,:);
            else            
                stcom = (i-1)*pieceheight - (i-1)*2*pieceoverlap + 1;
                edcom = i*pieceheight - (i-1)*2*pieceoverlap;
                stsam = (i-1)*pieceheight - ((i-1)*2-1)*pieceoverlap + 1;
                edsam = i*pieceheight - ((i-1)*2+1)*pieceoverlap;
                disp(['stsam: ' num2str(stsam)]);
                disp(['edsam: ' num2str(edsam)]);
                disp(['stsam: ' num2str(1+pieceoverlap)]);
                disp(['edsam: ' num2str(pieceheight-pieceoverlap)]);

                % apply the scalar on individual spectral channels
%                 scalar = scalarmat(i,:);
%                 h = size(x_twist,1);
%                 w = size(x_twist,2);
%                 d = size(x_twist,3);
%                 x_twist = reshape(x_twist,[h*w, d]);
%                 scalar = repmat(scalar,[h*w 1]);
%                 x_twist = scalar.*x_twist;
%                 x_twist = reshape(x_twist,[h,w,d]);

                tx_twist(stsam:edsam, :, :) = x_twist(1+pieceoverlap:(pieceheight-pieceoverlap),:,:);
            end            
            %disp(['scalars: ' num2str(scalarmat(i,:))]);
        end
    end

    %================================================
    % Apply CASSI characterization model
    %================================================
% if strcmp(type,'vis')    
%      x_twist = applycharacterization(tx_twist); % only for [450,464,480,497,516,537,560,586,614,645;] % floored
% end
    %x_twist = tx_twist;
    %{    
    %========================================================
    % I changed acquisition software (no need to flip anymore)
    %========================================================
    % flip left-right as CASSI data is flipped
    for i=1:size(x_twist,3)
        temp(:,:) = x_twist(:,:,i);
        x_twist(:,:,i) = fliplr(temp);
    end
%}

    %----------------------------------------------------------%
    % calibrating dispersion offset for UV and IR channels
    % note that this calibration depends on the input paramters
    % if the input parameters are changed, we have to recalibrate the
    % offsets of dispersion - Min (11-14-2011)
    % DON'T CHAGE THIS.
    if strcmp(type,'uv')==1 % UV filter
       x_twist = circshift(tx_twist,[0 47]); % DON'T CHANGE THIS CALIBRATION
    elseif strcmp(type,'ir')==1 % IR filter
       x_twist = circshift(tx_twist,[0 -16]); 
    elseif strcmp(type,'nofilter')==1
       x_twist = circshift(tx_twist,[0 35]);
    else % this is for 'vis', the yellow filter %elseif  strcmp(type,'vis')==1
       x_twist = tx_twist;
    end
    %----------------------------------------------------------%
    
    % save
    fn = recfilelist{1};
    [pathstr, name, ext] = fileparts(fn);
    name = regexprep(name, 'recon_', 'final_','ignorecase');
    fn = [pathstr, '/', name, ext];
    
    totaltime = [formattime(tttime)];  
    save(fn,'shiftb','taub','piterb','iterationsb','wvls2b','deltab',...
        'x_twist','totalres','rect','tttime','totaltime');

    %================================================
    % Store data as image
    %================================================
    %cassidisplay(0, 0, fn);
    cassiwriteimages(0, 'color', 1, 0, fn);

%     delete_intermediates;
    
end

