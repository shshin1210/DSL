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
% function sRGB=R2sRGB(wvls, R, refwhite, islinear)
% wvls is a 1D vector of (wavelength)
% R is a 3D matrix of (x,y,wavelength)
function sRGB=R2sRGB(wvls, R, refwhite, islinear, isstretch)
    if nargin<4
        islinear = 0;
    end
    if nargin<5
        isstretch = 0;
    end
    
    % remove noise floor in solid-state signals
%    nf = 0.0031; % lowest signals in the 8 channels (UV) -> this is only for QSI
%     R = R - nf;
        
    if refwhite==0 % automatic white balance
        [Ref, WhiteRef] = Rad2Ref(R,refwhite);% automatic white balance
    else % manual white balance. -> 3D scanning
        % white reference coefficient 
        % (The Yale Graphics Spectralon, measured in 3/13/2012
        % (1) 97.81% --> Reference White
        % (2) 50.14%
        % (3) 26.58%
        % (4) 14.89%        
        refco = 0.9781;
        refwhite2 = refwhite./refco;

        % workflow
        % 1. radiance to reflectance (scaling by refwhite)
        Ref = Rad2Ref(R,refwhite2);
        WhiteRef = Rad2Ref(refwhite,refwhite2);
    end

    % 2. reflectance to CIEXYZ (2-degree, D50)
    illumin = 'd65';
    Ref_XYZ = Ref2XYZ(wvls, Ref, illumin);
    White_XYZ = Ref2XYZ(wvls, WhiteRef, illumin);
    disp(['white point [(' illumin ') ' num2str(White_XYZ) ']']);

    if refwhite==0 % automatic white balance (grayworld)
        mmean = mean(mean(Ref_XYZ));
%         mmean = mmean(:);
        maxY = max(max(Ref_XYZ(:,:,2)));
        Ref_XYZw = mmean.*maxY./mmean(1,1,2);
        rpmean = repmat(Ref_XYZw, [size(Ref_XYZ,1), size(Ref_XYZ,2), 1]);
        Ref_XYZw = Ref_XYZ./rpmean;
        isstretch = 1;
    else
        % 3. Von Kries White Balancing in D65
        White_XYZ = reshape(White_XYZ,[1,1,3]);
        White_XYZrep = repmat(White_XYZ,[size(Ref_XYZ,1), size(Ref_XYZ,2), 1]);
        Ref_XYZw = Ref_XYZ./White_XYZrep; % here 100 -> 1.0
    end
    
    % apply D65 white point for sRGB
    wpD65 = whitepoint('d65')./100;
    wpD65 = reshape(wpD65,[1,1,3]);
    wpD65 = repmat(wpD65, [size(Ref_XYZ,1), size(Ref_XYZ,2), 1]);
    Ref_XYZw = wpD65.*Ref_XYZw;

    % optional process (stretching tone). 
    % don't use this by default (only for the final screen view)
    if isstretch==1
        % normalization based on Y -> fluctuate depending on image properties
        minY = min(min(Ref_XYZw(:,:,2)));
        maxY = max(max(Ref_XYZw(:,:,2)));
        temp = repmat(minY,[size(Ref_XYZw,1),size(Ref_XYZw,2),3]);
        Ref_XYZw = (Ref_XYZw - temp)./(maxY - minY);
        Ref_XYZw(Ref_XYZw>1)=1;
        Ref_XYZw(Ref_XYZw<0)=0;
        % histogram stretching from 1% to 99% 
        Ref_XYZw = stretch_hist(Ref_XYZw, 0.1, 99.9);
    end
    
    % 3. CIEXYZ to sRGB (gamma = 2.2);
    if islinear==1
        sRGB = XYZ2sRGBlinear(Ref_XYZw);
    else
        sRGB = XYZ2sRGB(Ref_XYZw);
    end
end    
