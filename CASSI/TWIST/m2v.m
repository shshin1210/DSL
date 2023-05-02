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
% function v = m2v(data)
function v = m2v(data)
% M2V: Converts rxcxn matrix of colour data to n columns. 
% The order of the data in the resulting columns is r1,r2,...rn
%
% Example: M2V(CMY) where CMY is a 3D matrix 
% arranged row x column x colour.
%
%   Colour Engineering Toolbox
%   author:    ?Phil Green
%   version:   1.1
%   date:  	   17-01-2001
%   book:      http://www.wileyeurope.com/WileyCDA/WileyTitle/productCd-0471486884.html
%   web:       http://www.digitalcolour.org

% Input matrix and get size
    LMN=data;
    [r,c,n]=size(LMN);

    % Prepare empty matrix
    v=zeros(r*c,n);

    % Transpose and reshape to vectors
    for i=1:n
       L=LMN(:,:,i);
       Lt=L';
       v(:,i)=Lt(:);
    end
end