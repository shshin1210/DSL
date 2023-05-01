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
% function sRGB = spectrumRGB(lambdaIn, varargin)
function sRGB = spectrumRGB(lambdaIn, varargin)
%spectrumRGB   Converts a spectral wavelength to RGB.
%
%    sRGB = spectrumRGB(LAMBDA) converts the spectral color with wavelength
%    LAMBDA in the visible light spectrum to its RGB values in the sRGB
%    color space.
%
%    sRGB = spectrumRGB(LAMBDA, MATCH) converts LAMBDA to sRGB using the
%    color matching function MATCH, a string.  See colorMatchFcn for a list
%    of supported matching functions.
%
%    Note: LAMBDA must be a scalar value or a vector of wavelengths.
%
%    See also colorMatchFcn, createSpectrum, spectrumLabel.

%    Copyright 1993-2005 The MathWorks, Inc.

if (numel(lambdaIn) ~= length(lambdaIn))
    
    error('spectrumRGB:unsupportedDimension', ...
          'Input must be a scalar or vector of wavelengths.')
    
end

% Get the color matching functions.
if (nargin == 1)
    
%    matchingFcn = '1931_full'; % CIE 2 degree
%    matchingFcn = '1964_full'; % CIE 10 degree
    matchingFcn = 'judd_vos'; % modified 2 degree
    
elseif (nargin == 2)
    
    matchingFcn = varargin{1};
    
else
    
    error(nargchk(1, 2, nargin, 'struct'))
    
end

[lambdaMatch, xFcn, yFcn, zFcn] = colorMatchFcn(matchingFcn);

% Interpolate the input wavelength in the color matching functions.
XYZ = interp1(lambdaMatch', [xFcn; yFcn; zFcn]', lambdaIn, 'pchip', 0);

% Reshape interpolated values to match standard image representation.
if (numel(lambdaIn) > 1)
    
    XYZ = permute(XYZ', [3 2 1]);
    
end

% Convert the XYZ values to sRGB.
XYZ2sRGB = makecform('xyz2srgb');
sRGB = applycform(double(XYZ), XYZ2sRGB);
