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

% normalize cassi data
%x_twist3= (x_twist - min(min(min(x_twist))))/(max(max(max(x_twist))) - min(min(min(x_twist))));

x_twist3= x_twist/max(max(max(x_twist))); % following David -> just trucate

x_twist3(x_twist3<0) = 0;
x_twist3(x_twist3>1) = 1;

% save them
for i=1:size(x_twist,3)
    gc = 1/2.2;
    aimg = x_twist3(:,:,i);
    aimg = fliplr(aimg); % flip images
    aimg = flipud(aimg); % flip images
    imshow(aimg);
    % filename
    fn=['results_' num2str(round(wvls2b(i))) 'nm.tif'];
    %imwrite(uint16(round((aimg.^gc)*65535)),fn,'tif'); % gamma correction
    imwrite(uint16(round(aimg*65535)),fn,'tif'); % gamma correction
end