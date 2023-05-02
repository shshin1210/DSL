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
% function offsetcorrect(inputfn)
function offsetcorrect(inputfn)
% extra final calibration for siggraph --> THIS IS WRONG (NOT NEEDED)
% -12/30/2011
    if nargin<1
        % file read
        [file_name,file_path] = uigetfile({'*.mat','mat';},'Choose a reconstructed file');
        % leave function when no file is selected.
        if size(file_path,2) == 1
            return;
        end
        inputfn = char(strcat(file_path,file_name));   
    end

    disp 'Reading data...';
    load(inputfn);
    disp 'Done.';

    %----------------------------------------------------------%    
    % extra final calibration for siggraph
    for ind=1:size(x_twist,3)
        % new finding (12/29/2011) if this is under 400, it should be
        % shifted to the left.
        if wvls2b(ind)<=405 %under 405nm
            sshift = -25;
            x_twist(:,:,ind)=circshift(x_twist(:,:,ind),[0 sshift]);
        elseif wvls2b(ind)>=980
            sshift = 8;
            x_twist(:,:,ind)=circshift(x_twist(:,:,ind),[0 sshift]);
        end
        % only for debug
        %disp([num2str(ind) '(' num2str(wvls2b(ind))  '): ' num2str(sshift)]);
        %imwrite(cal2(:,:,ind)./max(max(cal2(:,:,ind))),[num2str(ind) '.png']);
    end
    %----------------------------------------------------------%  
            
    disp 'Saving data...';
    outfn = regexprep(inputfn, 'FINAL_', 'FINAL_OFFSETC_','ignorecase');
    %outfn = [inputfn '_offsetcorrect.mat'];
    offset = ['This is offset corrected'];
    save(outfn,'wvls2b','x_twist','totalres','rect','offset');

    