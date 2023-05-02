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
% Acknowlegments
% Portions of this file were based on the original code of David Kittle 
% (Duke University).
%============================================================================
% function shiftedcube = shiftCube(unshiftedcube)
function shiftedcube = shiftCube(unshiftedcube, shifti)

if nargin < 2
    % shift -1 (in shift(1)) is to move 1pixel to the right
    % in Min's setting
    %shifti = -1; %DUKE
    shifti = -3; % this works fine with deltab = 3 FINAL
    %shifti = -2; %Min this works with deltab = 2
end

shiftedcube = zeros(size(unshiftedcube));

for i = size(unshiftedcube,3):-1:1
    shiftedcube(:,:,i) = circshift(unshiftedcube(:,:,i),[0 shifti*i]);
end