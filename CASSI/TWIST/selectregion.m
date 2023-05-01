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
% Display cassi data
% function [X, rect] = selectregion(S)
function [X, rect] = selectregion(S)
% S is the input image
% X is the selected region
% R contains the region coordinates

figure; imshow(S); set(gcf,'name','Select Region');

[y2 rect]=imcrop(S,[]);
if isempty(rect)
    sclr = 0;
    return;
end

% boolean mask
MASK = zeros([size(S,1) size(S,2)], 'uint8');

%rect=round(rect);
rect=floor(rect);
MASK(rect(2):rect(2)+rect(4) , rect(1):rect(1)+rect(3)) = 1; % 600-by-900

% disp(['========================================']);
% disp(['[Selected region]']);
% disp(['Rect (x,y,w-1,h-1):' num2str(uint16(rect))]);

[X, R] = imstack2vectors(S, MASK);
fr = R(end,:) - R(1,:) + [1 1];
r1 = reshape(X(:,1), fr(1), fr(2));

%figure;imshow(r1);set(gcf,'name','Selected Region');

% place the region image on the black mask
padding = MASK;
padding(rect(2):rect(2)+rect(4) , rect(1):rect(1)+rect(3)) = r1;

imshow(padding);set(gcf,'name','Zero-padded image');