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
% Convert Ocean Optics (USB2000, Yale EEB) measurements into regularized 1nm interval
% without calibration.
% How to use:
% 0. measure radiance with Ocean USB2000 (with "dark-current removal (S-)")
% 1. create a matrix of Ocean measurements (1st column: wavelengths, 2nd column: signals) as row vectors.
% this calibration can be used with any machines (pc or mac).
%==============================================================================================%
% function regsing = ocean2nm(signals)
function regsing = ocean2nm(signals)
    signals(:,1) = floor(signals(:,1));
    % regularizing ocean signals
    regsing = regularize(signals);       
end

% change the data interval and conduct averaging within the interval.
function regsign = regularize(signals,interval)

    if nargin<2
        interval=1;
    end

    signals(:,1) = signals(:,1)/interval;

    % check the number of bins
    count = size(signals,1);
    
    
    for j=2:size(signals,2)
        %---------------------------------------------------
        % initial run for the very first
        % read the current wavelength
        iwave = floor(signals(1,1));
        % read the current radiance
        irad = signals(1,j);
        prewave = iwave;
        isum = irad;
        k=1;
        icount = 1;
        %---------------------------------------------------
        for i=2:count
            % read the current wavelength
            iwave = floor(signals(i,1));
            % read the current radiance
            irad = signals(i,j);

            if iwave == prewave
                icount = icount + 1; 
                isum = isum + irad;
            else % beginning of the segment
                % porting the wavelength
                regsign(k,1) = prewave+1;
                regsign(k,j) = isum / icount;  % calculate average
                k = k + 1;
                isum = irad;
                icount = 1;
            end
            prewave = iwave;
        end
    end
    regsign(regsign<0)=0;
    
    regsign(:,1) = regsign(:,1)*interval;

end