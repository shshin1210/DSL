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
% function merge_uvvisir(fnuv,fnvis,fnir,nopad,isRad)
function merge_uvvisir(fnuv,fnvis,fnir,nopad,isRad)
    if nargin < 1
        %nopad = 1; % zero-padding off
        nopad = 0; % zero-padding on
    end
    if nargin < 2
        %isRad = 1; % radiance
        isRad = 0; % reflectance
    end
    if (nargin<3)
    % file read
    [file_name,file_path] = uigetfile({'*.mat','mat';},'Choose UV reconstruction');
    % leave function when no file is selected.
    if size(file_path,2) == 1
        return;
    end
    
    fnuv = char(strcat(file_path,file_name));   
    end
    disp (['loading: ' fnuv]);
    
    
    
    load(fnuv);
    
    ttime = tttime;
    
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
    
    %==================================================================% 
    % Combining UV with VIS
    %==================================================================% 
    % file read
    if (nargin<3)
    [file_name,file_path] = uigetfile({'*.mat','mat';},'Choose VIS reconstruction');
    % leave function when no file is selected.
    if size(file_path,2) == 1
        return;
    end
    fnvis = char(strcat(file_path,file_name));   
    end
    disp (['loading: ' fnvis]);
    load(fnvis);
    
    ttime = ttime + tttime;
    
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
    tx_twist = cat(3, uv_twist(:,:,1:end), vis_twist(:,:,2:end));
    twvls2b = cat(2, uv_wvls2b(:,1:end), vis_wvls2b(:,2:end));

    
    %==================================================================% 
    % Combining UV-VIS with IR
    %==================================================================% 
    % file read
    if(nargin<3)
    [file_name,file_path] = uigetfile({'*.mat','mat';},'Choose IR reconstruction');
    % leave function when no file is selected.
    if size(file_path,2) == 1
        return;
    end
    fnir = char(strcat(file_path,file_name));   
    end
    disp (['loading: ' fnir]);
    load(fnir);
    
    ttime = ttime + tttime;
    
    %rename
    ir_twist = x_twist;
    clear('x_twist');
    ir_wvls2b = wvls2b;
    clear('wvls2b');

    % concatinate spectrum
    % VIS:  ------    680 (better) |  692 (worse), 705 (bad)
    % IR:  668 (bad), 680 (worse)  |  692 (better) -------              
    x_twist = cat(3, tx_twist(:,:,1:(end-2)), ir_twist(:,:,3:(end-1)));
    wvls2b = cat(2, twvls2b(:,1:(end-2)), ir_wvls2b(:,3:(end-1)));

    %================================================
    %apply the radiometric calibration % 2011-11-21 SIGGRAPH
    %================================================
    x_twist = apply_3dis_calib(x_twist);       
    
    
    %================================================
    % Store data as image
    %================================================
%    sourcepath = regexprep(sourcepath, '.mat', '_UV-VIS.mat','ignorecase');
    [pathstr, name, ext] = fileparts(fnir);
    if isempty(pathstr)
        pathstr = '.';
    end
    name = regexprep(name, 'ir', 'All','ignorecase');
    
    sourcepath = [pathstr, '/', name, ext];
    totaltime = [formattime(ttime)]; 
    save(sourcepath,'wvls2b','x_twist','totalres','rect','totaltime');

    if (nargin<3)
        cassidisplay(0, 0, sourcepath);
    else
        cassiwriteimages(0,'color',1,0,sourcepath);
    end
end

