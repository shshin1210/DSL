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
% function cassirect(crect)
function cassirect(crect)
% chopping=800 --> create the mini color chart crop at the center
% We have to crop the edge trim of the coded aperture.

    if nargin<1
        crect = 0;
    end
    close all;
        
    file_name_main = [''];
    
    %=====================================================================
    % ask filename for the main listfile
    if (isempty(file_name_main))
        % file read
        [file_name_main,file_path_main] = uigetfile({'*.mat','mat';},'Choose a main capture mat file');
        % leave function when no file is selected.
        if size(file_path_main,2) == 1
            return;
        end
    else
        file_path_main = [pwd '/'];          
    end

    sourcepath = char(strcat(file_path_main,file_name_main));  
    
    % load the source MAT file
    load(sourcepath);

    % read the size of cdata
    if (isempty(I_trans))
        return;
    else
        height = size(I_trans,1);
        width = size(I_trans,2);
    end
    %=====================================================================
    % total resolution of one side of the square sensor
    totalres = size(I_trans,1);
    %=====================================================================
    % chopping edge pixels (removing the trim of coded aperture)
    % chopping 900 makes 249x249 resolution image. It takes only 4 minutes
    % default should be 50 in order to remove the black trim of the coded
    % aperature.
    chopping = 48; % DEFAULT should be 48
    I_trans = I_trans(chopping:size(I_trans,1)-chopping,chopping:size(I_trans,1)-chopping,:);
    
    % Chopping source edge data
    %img_c = uint8(cdata(:,:));
    img_i = uint8(I_trans(:,:,1));
    %figure; imshow(img_c); set(gcf,'name','Coded aperture');
    
    %=====================================================================
    % with default chopping
    % these rect values are locations in 2024 x 2024
    % launch the selection UI
    if crect==0
        [X crect] = selectregion(img_i);
    end
    rect(1) = crect(1) + chopping; % x
    rect(2) = crect(2) + chopping; % y
    rect(3) = crect(3); % width
    rect(4) = crect(4); % height

    % segment the data
    I_trans = I_trans(crect(2):crect(2)+crect(4) , crect(1):crect(1)+crect(3),:);

    %=====================================================================
    % After cropping the cdata and I_trans
    % we need to account for the chopping offsets in rect(1) and rect(2)
    disp(['========================================']);
    disp(['[Selected region in 2024x2024]']);
    disp(['Rect (x,y,w-1,h-1):' num2str(uint16(rect))]);
    %=====================================================================
    
    % show cropped results
    imshow(uint8(I_trans(:,:,1)));set(gcf,'name','Dispersion capture');
    %------------------------------------------------------------------

    %save( [sourcepath '_rect.mat'], 'rect');
    save( 'rect.mat', 'rect');
    
    return;
end
