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
% function R_XYZ=R2XYZ(wvls, R)
% wvls is a 1D vector of (wavelength)
% R is a 3D matrix of (x,y,wavelength)
function XYZ = Ref2XYZ(wvls, Ref, illum)	
	%maximum photographic luminous efficacy
	Km = 683;
	
    % derive color matching function
    %[lambdaCMF, xFcn, yFcn, zFcn] = colorMatchFcn('1931_full');
    [lambdaCMF, xFcn, yFcn, zFcn] = colorMatchFcn('judd_vos'); % this is closer to the eye

    [lambdaLumin, LFcn] = illuminantFcn(illum);

    waveArrayCMF = @(x) uint8(x - (lambdaCMF(1)-1));
    waveArrayLumin = @(x) uint8(x - (lambdaLumin(1)-1));

    if ndims(Ref)==3
        hh = size(Ref,1);
        ww = size(Ref,2);
        dd = size(Ref,3);

        Lybar = zeros(hh,ww,dd);
        XYZ = repmat(Ref, [1,1,1,3]);
%        XYZ = zeros(hh,ww,dd,3); % too much memory
        
        
        for i=1:size(Ref,3) % i is wavelength
            wvarrayCMF = waveArrayCMF(floor(wvls(i)));
            wvarrayLumin = waveArrayLumin(floor(wvls(i)));
            % fit the luminance range
            if floor(wvls(i)) < 380 || floor(wvls(i)) > 730
                XYZ(:,:,i,1) = 0.;
                XYZ(:,:,i,2) = 0.;
                XYZ(:,:,i,3) = 0.;
                % scalar L(lambda)*y(lambda)
                Lybar(:,:,i) = repmat(0.,[hh,ww]);
            else
                XYZ(:,:,i,1) = XYZ(:,:,i,1).*xFcn(wvarrayCMF).*LFcn(wvarrayLumin);
                XYZ(:,:,i,2) = XYZ(:,:,i,2).*yFcn(wvarrayCMF).*LFcn(wvarrayLumin);
                XYZ(:,:,i,3) = XYZ(:,:,i,3).*zFcn(wvarrayCMF).*LFcn(wvarrayLumin);
                % scalar L(lambda)*y(lambda)
                Lybar(:,:,i) = repmat(yFcn(wvarrayCMF).*LFcn(wvarrayLumin),[hh,ww]);
            end
        end
        
        % summation of energy (4D -> 3D)
        XYZ = sum(XYZ, 3);
        XYZ = reshape(XYZ, [size(Ref,1),size(Ref,2),3]);
        
        % normalization
        k = 100./sum(Lybar, 3);
        k(k==Inf) = 0;
        k = repmat(k,[1,1,3]);
        XYZ = k.*XYZ;

        % remove below zero
        XYZ(XYZ<0) = 0.0;
    elseif ndims(Ref)==2
        
        dd = size(Ref,1); % wavelength
        vv = size(Ref,2); % other points
        
        Lybar = zeros(dd,vv);
        XYZ = repmat(Ref, [1,1,3]);
%        XYZ = zeros(dd,vv); % too much memory
        
        for i=1:size(Ref,1) % i is wavelength
            wvarrayCMF = waveArrayCMF(floor(wvls(i)));
            wvarrayLumin = waveArrayLumin(floor(wvls(i)));
            % fit the luminance range
            if floor(wvls(i)) < 380 || floor(wvls(i)) > 730
                XYZ(i,:,1) = 0.;
                XYZ(i,:,2) = 0.;
                XYZ(i,:,3) = 0.;
                % scalar L(lambda)*y(lambda)
                Lybar(i,:) = repmat(0.,[1,vv]);
            else
                XYZ(i,:,1) = XYZ(i,:,1).*xFcn(wvarrayCMF).*LFcn(wvarrayLumin);
                XYZ(i,:,2) = XYZ(i,:,2).*yFcn(wvarrayCMF).*LFcn(wvarrayLumin);
                XYZ(i,:,3) = XYZ(i,:,3).*zFcn(wvarrayCMF).*LFcn(wvarrayLumin);
                
                % scalar L(lambda)*y(lambda)
                Lybar(i,:) = repmat(yFcn(wvarrayCMF).*LFcn(wvarrayLumin),[1,vv]);
            end
        end
        
        % summation of energy (3D -> 2D)
        XYZ = sum(XYZ, 1);
        XYZ = reshape(XYZ, [vv,3]);
        
        % normalization
        k = 100./sum(Lybar,1);
        k(k==Inf) = 0;
        k = repmat(k,[vv,3]);
        XYZ = k.*XYZ;

        % remove below zero
        XYZ(XYZ<0) = 0.0;
    else
        return;
    end