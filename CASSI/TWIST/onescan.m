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
% function onescan(arg, fn_calib, handles)
function onescan(arg, fn_calib, handles)
    prefix = [get(handles.edit_prefix,'String') '_'];
    
    fn_low3pi = [prefix 'low_' datestr(clock,30) '.3pi'];
    fn_high3pi = [prefix 'high_' datestr(clock,30) '.3pi'];
    fn_lowtif = regexprep(fn_low3pi, '.3pi', '.tif','ignorecase');
    fn_hightif = regexprep(fn_high3pi, '.3pi', '.tif','ignorecase');

    if strcmp(arg,'high')
        fn_3pi = fn_high3pi;
        fn_tif = fn_hightif;
    else
        fn_3pi = fn_low3pi;
        fn_tif = fn_lowtif;
    end        
        
    % !SgAcquire -s fnCalib  fnCalib
    cmd = ['SgAcquire -s ' fn_calib ' ' fn_3pi];
    [status,result] = system(cmd,'-echo');
    pause(2);
    %!tpi2tif high_laser.3pi high_laser.tif
    cmd = ['tpi2tif ' fn_3pi ' ' fn_tif ];
    [status,result] = system(cmd,'-echo');
    disp('Range scan done.');
    
    pause(20);
    img = 1 - double(imread(fn_tif))./255;
    figure;imshow(img);
end