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
% Write a multi-channel exr
% Linear albedo maps (No Gamma Correction)!!!
% Author:  (c) 2011 Min H. Kim (Yale University)
% Dependency: Edgar's OpenEXR library
%============================================================================
% function writeexr(filename)
function writeexr(filename)

    COMPRESSION = 'piz';
    PIXELTYPE = 'float';
    
    if nargin<1
        % file read
        [file_name,file_path] = uigetfile({'*.mat','mat';},'Choose a reconstructed file');
        % leave function when no file is selected.
        if size(file_path,2) == 1
            return;
        end
        sourcepath = char(strcat(file_path,file_name));   
    end
    if nargin<2
        correctgamma=1;
    end
    
    disp 'Reading data...';
    load(sourcepath);
    disp 'Done.';
    
    % channel naming
    for i=1:size(wvls2b,2)
        cst{i} = [num2str(floor(wvls2b(i))) 'nm'];
    end

    cNames = {'R', 'G', 'B', cst{:}}; 
    
    outpath = regexprep(sourcepath, '.mat','.exr','ignorecase');
    
    sRGB = cassidisplay(0, 1, sourcepath);
    
    %================================================
    % Store data as image
    %================================================
    % Gaussian blur (doesn't make big difference)
    %hsize = [3 3];
    %sigma = 0.5;
    %h = fspecial('gaussian', hsize, sigma); % not working (11/18/2011)
    h = fspecial('disk',10); % better for noise
    x_blurred = imfilter(x_twist,h,'replicate');

    xtwistmin = min(min(min(x_blurred)));
    xtwistmax = max(max(max(x_blurred)));    
%    xtwistmax = max(max(max(x_twist)));

    nx_twist = x_twist./xtwistmax;    

    
    cData{1} = single(sRGB(:,:,1));
    cData{2} = single(sRGB(:,:,2));
    cData{3} = single(sRGB(:,:,3));
    % channel naming
    for i=1:size(nx_twist,3)
        cData{i+3} = single(nx_twist(:,:,i)); 
    end
    
    exrwritechannels(outpath, COMPRESSION, PIXELTYPE, cNames, cData);
    
end