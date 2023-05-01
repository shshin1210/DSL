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
% function merge_uvvis(type,nopad)
function merge_uvvis(type,nopad)
    if nargin < 2
        nopad = 1; % zero-padding off
        %nopad = 0; % zero-padding on
    end
    
    if nargin==0;
       type = 'gray'; 
    end
    % file read
    [file_name,file_path] = uigetfile({'*.mat','mat';},'Choose UV reconstruction');
    % leave function when no file is selected.
    if size(file_path,2) == 1
        return;
    end
    sourcepath = char(strcat(file_path,file_name));   
    
    load(sourcepath);

    uv_twist = x_twist;
    %----------------------------------------------------------%
    % calibrating dispersion offset for BLUE and YELLOW channels
    % NB This part is moved to 'merge_segment.m' (11/16/2011)
%    uv_twist = circshift(x_twist,[0 47]);    % check this is correct (40)
    % the offset of the semrock 'BLUE' filter is 47, TESTED BY MIN (11-16-2011)
    %----------------------------------------------------------% 
    clear('x_twist');
    uv_wvls2b = wvls2b;
    clear('wvls2b');
    
    % file read
    [file_name,file_path] = uigetfile({'*.mat','mat';},'Choose VIS reconstruction');
    % leave function when no file is selected.
    if size(file_path,2) == 1
        return;
    end
    sourcepath = char(strcat(file_path,file_name));   
    
    load(sourcepath);
    
    %rename
    vis_twist = x_twist;
    clear('x_twist');
    vis_wvls2b = wvls2b;
    clear('wvls2b');
    
    %----------------------------------------------------------%
    % Automatic estimation of the uv scalar --> not working
    % calculate uv scalar
%     y = vis_twist(:,:,1);
%     A = uv_twist(:,:,end);
%     y = y(:);
%     A = A(:);
%     x = mean(y)/mean(A);
%     y2 = A*x;
%     disp(['scalar: ' num2str(x)]);
%     uv_twist = uv_twist.*x;
    %----------------------------------------------------------%
    % instead, I treat both channels equal.
    %----------------------------------------------------------%
    
    % concatinate spectrum
    % the beginning channel of VIS is not good (due to low transmittance)
%    x_twist = cat(3, uv_twist(:,:,1:(end-1)), vis_twist);
%    wvls2b = cat(2, uv_wvls2b(:,1:(end-1)), vis_wvls2b);
    % instead, use BLUE (UV) 516nm
    x_twist = cat(3, uv_twist(:,:,1:end), vis_twist(:,:,2:end));
    wvls2b = cat(2, uv_wvls2b(:,1:end), vis_wvls2b(:,2:end));
    
%    sourcepath = regexprep(sourcepath, '.mat', '_UV-VIS.mat','ignorecase');
    [pathstr, name, ext] = fileparts(sourcepath);
    name = regexprep(name, 'vis_', 'all_','ignorecase');
    sourcepath = [pathstr, '/', name, ext];
    save(sourcepath,'wvls2b','x_twist','totalres','rect');

    
    %================================================
    % Store data as image
    %================================================
    cassidisplay(0, 0, sourcepath);

end

