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
% function RGB=XYZ2sRGBlinear(XYZ0)
function RGB=XYZ2sRGBlinear(XYZ0)
    % XYZ2sRGB: calculates IEC:61966 sRGB values from XYZ
    % define 3x3 matrix
    % IEC61966-2-1
    M =[ 3.2406, -1.5372, -0.4986
        -0.9689,  1.8758,  0.0415
         0.0557, -0.2040,  1.0570];

    if ischar(XYZ0)
       XYZ=dlmread(XYZ0,'\t');
    elseif isnumeric(XYZ0)
       XYZ=XYZ0;
    else
       error('No valid input data')
    end

    % in the case of image
    if ndims(XYZ0)==3
        row=size(XYZ0,1);
        col=size(XYZ0,2);
        XYZ = reshape(XYZ0, row*col,3);
    end

    sRGB=(M*XYZ')';

    sR=sRGB(:,1);sG=sRGB(:,2);sB=sRGB(:,3);
    sR(sR>1)=1;sG(sG>1)=1;sB(sB>1)=1;
    sR(sR<0)=0;sG(sG<0)=0;sB(sB<0)=0;

    %apply gamma function
    %g=1/2.2;
    g=1.; % for linear gamma (EXR)
    
    % scale to range 0-255
    R=sR.^g;
    G=sG.^g;
    B=sB.^g;

    % clip to range
    R(R>1)=1;G(G>1)=1;B(B>1)=1;
    R(R<0)=0;G(G<0)=0;B(B<0)=0;
    
    RGB=[R,G,B];
    
    %if we want to 255 sRGB signals
    RGB = uint8(255.*RGB);

    if ndims(XYZ0)==3
        RGB=reshape(RGB, row,col,3);
    end
end