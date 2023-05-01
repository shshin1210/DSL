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
% function cf = smoothsplinefitting(pix,piy,smoothon)
function cf = smoothsplinefitting(pix,piy,smoothon)
    %CREATEFIT Create plot of data sets and fits
    %   CREATEFIT(PIX,PIY)
    %   Creates a plot, similar to the plot in the main Curve Fitting Tool,
    %   using the data that you provide as input.  You can
    %   use this function with the same data you used with CFTOOL
    %   or with different data.  You may want to edit the function to
    %   customize the code and this help message.
    %
    %   Number of data sets:  2
    %   Number of fits:  1

    % Data from data set "piy vs. pix":
    %     X = pix:
    %     Y = piy:
    if nargin < 3
        smoothon = 'on';
    end
    pix = pix(:);
    piy = piy(:);

    % smoothing piy
    sigma = 0.25; % [default] higher means smoother
    piysm = smooth(pix,piy,sigma,'loess',0); % dont use 'rloess' -> offsets
    
    % --- Create fit "fit 1"
    ok = isfinite(pix) & isfinite(piy);
    if ~all( ok )
        warning( 'GenerateMFile:IgnoringNansAndInfs',...
            'Ignoring NaNs and Infs in data.' );
    end
    ft = fittype('smoothingspline');

    warning off;
    % Fit this model using new data
    if strcmp(smoothon,'on')
        cf = fit(pix(ok),piysm(ok),ft); % smoothed fitting
    else
        cf = fit(pix(ok),piy(ok),ft); % raw fitting
    end
end
