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
% function wbR = R2WBR(R,refwhite,isauto)
% wvls is a 1D vector of (wavelength)
% R is a 3D matrix of (x,y,wavelength)
function [wbR, refwhite] = Rad2Ref(R,refwhite)%,isauto)
minimum = 0.0;
R(R<minimum) = minimum;

h = size(R,1);
w = size(R,2);
d = size(R,3);
if nargin<2
    refwhite=0;
end
if nargin<3
    isauto=0;
end

if (size(refwhite,1)==1 && refwhite(1,1)==0)
    wbR = R;
    return;
else
    if (ndims(R)==3)
        temp = reshape(refwhite, [1,1,d]);
        temp = repmat(temp, [h,w,1]);
        wbR = R./temp;
        return;
    elseif (ndims(R)==2)
        wbR = R./refwhite;
        return;
    else
        return;
    end
end