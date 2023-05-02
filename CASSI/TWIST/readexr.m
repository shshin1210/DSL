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
%==================================================%
% Read a multi-channel exr
% Linear albedo maps (No Gamma Correction)!!!
% Dependency: Edgar's OpenEXR library
%==================================================%
% function [sRGB, HCH, Names] = readexr(filename,channel)
function [sRGB, HCH, Names] = readexr(filename,channel)

    if nargin<1
        % file read
        [file_name,file_path] = uigetfile({'*.exr','exr';},'Choose a reconstructed file');
        % leave function when no file is selected.
        if size(file_path,2) == 1
            return;
        end
        sourcepath = char(strcat(file_path,file_name));   
        mapObj = exrreadchannels(sourcepath);
    elseif nargin<2
        sourcepath = filename;   
        mapObj = exrreadchannels(sourcepath);
    elseif nargin==2
        % file read
        [file_name,file_path] = uigetfile({'*.exr','exr';},'Choose a reconstructed file');
        % leave function when no file is selected.
        if size(file_path,2) == 1
            return;
        end
        sourcepath = char(strcat(file_path,file_name));  
        mapObj = exrreadchannels(sourcepath,channel);
        HCH = mapObj;
        mapObj = mapObj.^(1/2.2);
        figure; imshow(mapObj./max(max(mapObj)));  
        sRGB = 0;
        cNames = channel;
        return;
    end
    
    
    % note that order is changed (reverse order)
    cNames = keys(mapObj);
    cData = values(mapObj);
    
   % sRGB
    sRGB(:,:,1) = mapObj('R');
    sRGB(:,:,2) = mapObj('G');
    sRGB(:,:,3) = mapObj('B');
    Names{1} = 'R';
    Names{2} = 'G';
    Names{3} = 'B';
    
    srgb = sRGB.^(1/2.2);
    figure; imshow(srgb./max(max(max(srgb))));
    
    HCH = zeros([size(cData{1},1),size(cData{1},2),size(cData{1},3)-3]);
    
    % ORDER (sorted in writing): {'1002nm','359nm','364nm','369nm','375nm','382nm','389nm','397nm','405nm','415nm','425nm','437nm','450nm','464nm','480nm','497nm','516nm','523nm','530nm','537nm','544nm','552nm','560nm','568nm','577nm','586nm','595nm','604nm','614nm','624nm','635nm','645nm','657nm','668nm','680nm','692nm','705nm','718nm','732nm','746nm','761nm','776nm','791nm','807nm','824nm','841nm','859nm','878nm','897nm','916nm','937nm','958nm','980nm','B','G','R';}
    for i=1:(size(cData,2)-4)
        HCH(:,:,i) = single(cData{1+i});
        Names{i+3} = cNames{1+i};
    end
    HCH(:,:,size(cData,2)) = single(cData{1});
    Names{size(cData,2)} = cNames{1};
end