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
% function y_twist = kg3trans(wvls,x_twist)
function y_twist = kg3trans(wvls,x_twist)
    % coefficients
    a1 =      0.8563;%  (0.6367, 1.076)					
    b1 =       330.4;%  (271.3, 389.6)					
    c1 =       228.9;%  (-355.8, 813.6)					
    a2 =      0.6402;%  (-0.7989, 2.079)					
    b2 =       604.1;%  (532.5, 675.6)					
    c2 =       141.5;%  (72.49, 210.4)					
    a3 =     0.04258;%  (-0.004177, 0.08933)					
    b3 =       482.8;%  (469.9, 495.6)					
    c3 =       39.12;%  (11.51, 66.74)	

    fx = a1.*exp(-((wvls-b1)./c1).^2) + a2.*exp(-((wvls-b2)./c2).^2) + a3.*exp(-((wvls-b3)./c3).^2);
    rc_fx = 1./fx;
    for i=1:size(rc_fx,2)
        scalr(:,:,i) = repmat(rc_fx(i),[size(x_twist,1) size(x_twist,2)]);
    end
    y_twist = scalr.*x_twist;
    
%----------------------------------------------------------------------
% Characterization model (FINAL)
%     Based on 360-800nm					
%     General model Gauss3:					
%          f(x) = 					
%                   a1*exp(-((x-b1)/c1)^2) + a2*exp(-((x-b2)/c2)^2) + 					
%                   a3*exp(-((x-b3)/c3)^2)					
%     Coefficients (with 95% confidence bounds):					
%            a1 =      0.8563  (0.6367, 1.076)					
%            b1 =       330.4  (271.3, 389.6)					
%            c1 =       228.9  (-355.8, 813.6)					
%            a2 =      0.6402  (-0.7989, 2.079)					
%            b2 =       604.1  (532.5, 675.6)					
%            c2 =       141.5  (72.49, 210.4)					
%            a3 =     0.04258  (-0.004177, 0.08933)					
%            b3 =       482.8  (469.9, 495.6)					
%            c3 =       39.12  (11.51, 66.74)	
%     Goodness of fit:		
%       SSE: 0.3926		
%       R-square: 0.9869		
%       Adjusted R-square: 0.9867		
%       RMSE: 0.03015		
%----------------------------------------------------------------------
end