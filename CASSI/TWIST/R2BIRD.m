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
% function R_USML=R2BIRD(wvls, R)
% wvls is a 1D vector of (wavelength)
% R is a 3D matrix of (x,y,wavelength)
function R_USML=R2BIRD(wvls, R)
    % derive color matching function
    M = usmlAvgBird();
    % lambda starts from 300nm to 700nm
    % 1st array = 300nm
    % so, wavelength -> array number  <=> array_num = wavelength - 299
    waveArray = @(x) uint8(x - 299);

    % wavelength -> XYZ
    R_USML = repmat(R, [1,1,1,4]);
    % below 400nm -> wvarray (7); above 700nm -> wvarray (36)
    for i=1:size(R,3)
        wvarray = waveArray(floor(wvls(i)));
      % (Min) To avoid cyan and reddish color at the final
        if floor(wvls(i)) < 300 || floor(wvls(i)) > 700
            R_USML(:,:,i,1) = R_USML(:,:,i,1).*0.;
            R_USML(:,:,i,2) = R_USML(:,:,i,2).*0.;
            R_USML(:,:,i,3) = R_USML(:,:,i,3).*0.;
        else
            R_USML(:,:,i,1) = R_USML(:,:,i,1).*M(wvarray,2);
            R_USML(:,:,i,2) = R_USML(:,:,i,2).*M(wvarray,3);
            R_USML(:,:,i,3) = R_USML(:,:,i,3).*M(wvarray,4);
            R_USML(:,:,i,4) = R_USML(:,:,i,4).*M(wvarray,5);
        end
    end

    % summation of energy (4D -> 3D)
    R_USML = sum(R_USML, 3); % only 3th channel summ ( wave -> 4 cones)
    R_USML = reshape(R_USML, [size(R,1),size(R,2),4]);
    R_USML(R_USML<0) = 0.0;