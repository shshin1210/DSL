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
% function a = curvefitzero(x,y)
function a = curvefitzero(x,y)
x = x(:);
y = y(:);

% --- Create fit "fit 1"
ok_ = isfinite(x) & isfinite(y);
if ~all( ok_ )
    warning( 'GenerateMFile:IgnoringNansAndInfs',...
        'Ignoring NaNs and Infs in data.' );
end

% model equation: y = a*x
ft_ = fittype({'x'},...
    'dependent',{'y'},'independent',{'x'},...
    'coefficients',{'a'});

% Fit this model using new data
[cf_, gof_] = fit(x(ok_),y(ok_),ft_);

a = cf_.a;