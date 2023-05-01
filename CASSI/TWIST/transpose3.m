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
% transpose 3 dimensional
% function mat2=transpose3(mat)
function mat2=transpose3(mat)
    m1=mat(:,:,1);
    m2=mat(:,:,2);
    m3=mat(:,:,3);
    
    m1a=transpose(m1);
    m2a=transpose(m2);
    m3a=transpose(m3);
    
    mat2(:,:,1)=m1a;
    mat2(:,:,2)=m2a;
    mat2(:,:,3)=m3a;
    