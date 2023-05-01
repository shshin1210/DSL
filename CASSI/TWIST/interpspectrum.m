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
function [outimg, outwav] = interpspectrum(inimg, inwav, type)

switch type
    default:
    case 'interp':
        x = inwav;
        xi = [inwav(1):10:inwav(end)];
        y = inimg;

        hh = size(inimg,1);
        ww = size(inimg,2);
        dd = size(xi,2);

        for i=1:hh
            for j=1:ww
                temp = y(i,j,:);
                yi(i,j,:) = interp1(x,temp(:),xi);
            end
        end
    
        outwav = xi;
        outimg = yi;
        break;
    case 'qsi8':
        break;
    case 'qsi5':
        break;
end