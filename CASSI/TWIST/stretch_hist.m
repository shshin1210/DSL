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
% function [RGB2] = stretch_hist(RGB);
function [RGB2] = stretch_hist(RGB,LOW,HIGH);
if nargin<2
    % input 0to1
    HIGH = 99.9;
    LOW = 0.1; 
end
    RGB = single(RGB);
    
    %%[c1,v1]=probhist(RGB);
    pcnt_high = HIGH / 255.0;
    pcnt_low = LOW / 255.0;

    RGB2 = single(min(RGB,pcnt_high)); % returns the smallest elements than high
    RGB2 = single(max(RGB2,pcnt_low)); % returns the largest elements than low
    
    clear('RGB');
    
    % re_normalizing
    RGB2 = (RGB2-pcnt_low)./(pcnt_high-pcnt_low);
    
    RGB2(RGB2>1.0) = 1.0;
    RGB2(RGB2<0.0) = 0.0;
    
%     [c2,v2]=probhist(RGB2);
%     figure;imshow(RGB2);
%     figure;bar([c2(:,3),c2(:,2),c2(:,1)]);
    
