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
% function delete_intermediates
function delete_intermediates
    if exist('filelist_main.txt','file')
        mainfilelist = textread('filelist_main.txt', '%s');
    end
    if exist('filelist_cal.txt','file')
        califilelist = textread('filelist_cal.txt', '%s');
    end
    if exist('filelist_rec.txt','file')
        recfilelist = textread('filelist_rec.txt', '%s');
    end
    if exist('vis_reconfilelist.txt','file')
        vis_reconfilelist = textread('vis_reconfilelist.txt', '%s');
    end
    if exist('uv_reconfilelist.txt','file')
        uv_reconfilelist = textread('uv_reconfilelist.txt', '%s');
    end
    disp 'Reading lists done!';


    % delete files
     if exist('filelist_main.txt','file')
     inum = size(mainfilelist, 1);
     for i=1:inum
         delete(mainfilelist{i});
         disp ([mainfilelist{i} ' deleted']);
     end        
     end
    if exist('filelist_cal.txt','file')
    inum = size(califilelist, 1);        
    for i=1:inum
        delete(califilelist{i});
        disp ([califilelist{i} ' deleted']);
    end        
    end
    if exist('filelist_rec.txt','file')
    inum = size(recfilelist, 1);          
    for i=1:inum
        delete(recfilelist{i});
        disp ([recfilelist{i} ' deleted']);
    end
    end
    
    if exist('vis_reconfilelist.txt','file')
    inum = size(vis_reconfilelist, 1);          
    for i=1:inum
        delete(vis_reconfilelist{i});
        disp ([vis_reconfilelist{i} ' deleted']);
    end
    end
    
    if exist('uv_reconfilelist.txt','file')
    inum = size(uv_reconfilelist, 1);          
    for i=1:inum
        delete(uv_reconfilelist{i});
        disp ([uv_reconfilelist{i} ' deleted']);
    end
    end
    
    delete('cdata.bin');
    delete('Cu.bin');
    delete('obj_twist.bin'); 
    delete('shift.bin');
    delete('param.txt');
    delete('x_twist.bin');
    delete('twistoutput.txt');
    delete('timeshit.mat');
    delete('filelist_*.txt')
    delete('vis_reconfilelist.txt');
    delete('uv_reconfilelist.txt');
end