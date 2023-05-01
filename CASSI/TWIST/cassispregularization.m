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
% function cassispregularization(type, sourcepath)
function cassispregularization(type, sourcepath)
t0 = tic;
%load VIS_RAW_20110121T164339_rec_denoise.mat;
load test.mat;
disp(['reading input data.']); 

w = size(x_twist,2);
h = size(x_twist,1);

wvls2 = [400:10:700];
x_twist2 = zeros(h,w,size(wvls2,2));

disp(['regularizing the samples...']); 

% operation will be pixel-by-pixel.
parfor i=1:w
    disp(['row: ' num2str(i)]); 
    t1 = tic;
    for j=1:h
        piy(1,:) = x_twist(j,i,:);
        pix(1,:) = wvls2b(1,:);
        
        cf = smoothsplinefitting(pix,piy);
        
        for k=1:size(wvls2,2)
            x_twist2(j,i,k) = cf(wvls2(k));
        end
        
    end
    toc(t1);
end

x_twist = x_twist2;
clear('x_twist2');

wvls2b = wvls2;
clear('wvls2');
disp(['writing input data.']); 
save('regularized_data.mat', 'x_twist', 'wvls2b');
disp(['done.']); 
toc(t0);
end


function cf_ = smoothsplinefitting(pix,piy)
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

pix = pix(:);
piy = piy(:);

sm_.y2 = smooth(pix,piy,0.25,'loess',0);


% --- Create fit "fit 1"
ok_ = isfinite(pix) & isfinite(piy);
if ~all( ok_ )
    warning( 'GenerateMFile:IgnoringNansAndInfs',...
        'Ignoring NaNs and Infs in data.' );
end
ft_ = fittype('smoothingspline');

warning off;
% Fit this model using new data
cf_ = fit(pix(ok_),piy(ok_),ft_);
end
