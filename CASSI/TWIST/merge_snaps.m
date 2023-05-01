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
% function merge_snaps(file_name1, file_name2)
function merge_snaps(file_name1, file_name2)
    if nargin == 0
        file_name1 = '';
        file_name2 = '';
    end
    if (isempty(file_name1))
        % file read
        [file_name1,file_path] = uigetfile({'*.mat','mat';},'Choose a series of snaps');
        % leave function when no file is selected.
        if size(file_path,2) == 1
            return;
        end
    else
        file_path = [pwd '/'];
    end
    sourcepath1 = char(strcat(file_path,file_name1)); 
    disp(['loading: ' sourcepath1]);
    load(sourcepath1);
    
    %rename
    I_trans1 = I_trans;
    clear('I_trans');
    shift1 = shift;
    clear('shift');
    total_time1 = total_time;
    clear('total_time');
    
    if (isempty(file_name2))
        % file read
        [file_name2,file_path] = uigetfile({'*.mat','mat';},'Choose another series of snaps');
        % leave function when no file is selected.
        if size(file_path,2) == 1
            return;
        end
    else
        file_path = [pwd '/'];        
    end
    sourcepath2 = char(strcat(file_path,file_name2));  
    disp(['loading: ' sourcepath2]);
    load(sourcepath2);
    
    %rename
    I_trans2 = I_trans;
    clear('I_trans');
    shift2 = shift;
    clear('shift');
    total_time2 = total_time;
    clear('total_time');
   
    % concatinate spectrum
    I_trans = cat(3, I_trans1(:,:,1:end), I_trans2(:,:,2:end)); % start is overlaped
    shift = cat(2, shift1(:,1:end), shift2(:,2:end));
    total_time = total_time1 + total_time2;
    clear('I_trans1');    
    clear('shift1');
    clear('total_time1');
    clear('I_trans2');    
    clear('shift2');
    clear('total_time2');
    
    %sourcepath = regexprep(sourcepath1, 'raw_', 'merged_','ignorecase');
    [pathstr, name, ext] = fileparts(sourcepath1);
    name = regexprep(name, 'raw_', 'merged_','ignorecase');
    sourcepath = [pathstr, '/' name, ext];
    
    disp(['saving: ' sourcepath]);
    save(sourcepath,'I_trans','shift','total_time','mask_pix_size','shutters','ex_time','ex_gain','ex_stops','ex_num');
    %disp(['done! ']);
    disp(['deleting: ' sourcepath1]);
    delete(sourcepath1);
    disp(['deleting: ' sourcepath2]);
    delete(sourcepath2);
end

