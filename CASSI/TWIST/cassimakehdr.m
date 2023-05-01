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
% Acknowlegments
% Portions of this file were based on the original code of David Kittle 
% (Duke University).
%============================================================================
% function hdr = cassimakehdr(sourceimages,shutters, fstop, minclip, maxclip)
function hdr = cassimakehdr(sourceimages,shutters, fstop, minclip, maxclip)
    % instead of filenames, we use a multi-channel matrix (with exposure
    % variation), which should be raw linear signals.
    % here we are taking only an image matrix and relative exposure times array.

    % instead of reading information, from the file, we use direct input here.
    if(nargin<3)
        fstop=11;
    end
    
    if(nargin<4)
        % DONT INCLUDE THE LAST SIGNALS WE DON'T KNOW WHAT IT IS...
        minclip = 5;
        maxclip = 250;
    end
    
    disp(['Building HDR radiance map...']);
    
    % find out the base (minimum exposure energy)
    %[baseTime, baseFStop] = getExposure(options.basefile);
    baseFStop = fstop;
    baseTime = shutters(1); % it should be shotest shutter time. 


    height = size(sourceimages, 1);
    width = size(sourceimages, 2);

    [hdr, properlyExposedCount] = makeContainers(height, width);

    someUnderExposed = false(size(hdr));
    someOverExposed = false(size(hdr));
    someProperlyExposed = false(size(hdr));

    w = gweighting();
    
    for p = 1:size(sourceimages,3)
        disp(['exposure: ' num2str(p)]);
        
        this_FNumber = fstop;
        this_ExposureTime = shutters(p);
        
        relExposure = computeRelativeExposure(baseFStop, ...
                                              baseTime, ...
                                              this_FNumber, ...
                                              this_ExposureTime);

        % Read the LDR image
        ldr = sourceimages(:,:,p);

        underExposed = ldr < minclip;
        someUnderExposed = someUnderExposed | underExposed;

        overExposed = ldr > maxclip;
        someOverExposed = someOverExposed | overExposed;

        properlyExposed = ~(underExposed | overExposed);
        someProperlyExposed = someProperlyExposed | properlyExposed;

        properlyExposedCount(properlyExposed) = properlyExposedCount(properlyExposed) + 1;

        ldr(~properlyExposed) = 0;

%--------------------------------------------------------------
% DONT USE THIS WEIGHTING (LOSE BRIGHT PARTS DETAILS)
%         for j=1:size(ldr,1)
%             for k=1:size(ldr,2)
%                 wij(j,k) =  w(ldr(j,k)+1);
%             end
%         end
%         hdr = hdr + (wij.* single(ldr)) ./ (wij.*relExposure); % with weighting
%--------------------------------------------------------------
       
        hdr = hdr + single(ldr) ./ relExposure;
    end

    hdr = hdr ./ max(properlyExposedCount, 1);

    hdr(someOverExposed & ~someUnderExposed & ~someProperlyExposed) = max(hdr(someProperlyExposed));

    hdr(someUnderExposed & ~someOverExposed & ~someProperlyExposed) = min(hdr(someProperlyExposed));

    fillMask = imdilate(someUnderExposed & someOverExposed & ~someProperlyExposed, ones(3,3));
    if any(fillMask(:))
        hdr(:,:) = roifill(hdr(:,:), fillMask(:,:));
    end
end

function [hdr, counts] = makeContainers(height, width)
hdr = zeros(height, width, 'single');
counts = zeros(height, width,'single');
end


function relExposure = computeRelativeExposure(f1, t1, f2, t2)
relExposure = (f1 / f2)^2 * (t2 / t1);
end

% Computed equations:
% w(Zij) = Zij - Zmin for Zij<=(Zmax+Zmin)/2
% w(Zij) = Zmax - Zij for Zij> (Zmax+Zmin)/2
%

function w = gweighting
    zmin = 0;
    zmax = 255;
    zmid = 127;
    z=[0:1:255];

    for i=1: size(z,2)
        zij = z(1,i);
        if zij <= zmid
            wij = zij - zmin;
        else    %zij > zmid
            wij = zmax - zij;
        end
        w(i) = wij; %0-127
        w(i) = double(wij)/127; %0-1 normalization
    end
    % cover divided-by-zero error for channel builder
    w(1)=w(2);
    w(256)=w(255);
end