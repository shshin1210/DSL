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
% function [R, G, B, rgb] = readconvert3ch(filepath,bresize)
function [R, G, B, rgb] = readconvert3ch(filepath,bresize)
    if nargin<2
        bresize = 0;
    end
    % paramters
    gamma = 2.2;
    depth = 255.;

    rgborg = imread(filepath);
    
    if bresize == 1
    % this is for texture maps
    % find max size
    ysz = size(rgborg,1);
    xsz = size(rgborg,2);
    if rem(ysz,4) == 0
        imax = ysz;
    elseif rem(xsz,4) == 0
        imax = xsz;
    else
        imax = max(max(size(rgborg)));
    end
    
    % make it square
    isize = [imax imax]; % should be four-times number
    % resize images
    rgb = imresize(rgborg, isize, 'nearest');    
 
    else
        rgb = rgborg;
    end
    
    lrgb = (double(rgb)./depth).^gamma;

    R = lrgb(:,:,1);
    G = lrgb(:,:,2);
    B = lrgb(:,:,3);
end