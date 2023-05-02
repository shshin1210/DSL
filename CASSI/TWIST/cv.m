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
% function [CV,stdv] = cv(x,y)
function [CV,stdv] = cv(x,y)
    x = x(:);
    y = y(:);
    diffsq = (x-y).*(x-y);
    stdv = std(x,y);
    CV = 100/mean(y)*sqrt(sum(diffsq)./size(y,1));
end
    
    
    