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
% Convert Ocean Optics (USB2000, Yale EEB) measurements into calibrated radiance
% version 4.1
% in this version, Ocean is calibrated with GretagMacbeth EyeOne
% NOTE EYEONE SPECTRUM E IS 1,000 TIMES HIGHER THAN JETI SPECBOS

% How to use:
% 0. measure radiance with Ocean USB2000 (with "dark-current removal (S-)")
% 1. create a matrix of Ocean measurements (1st column: wavelengths, 2nd column: signals) as row vectors.
% 2. run "radiance = ocean2rad(signals);"
% 3. use the converted radiance values [unit: W/(sr*sqm*nm)]
% this calibration can be used with any machines (pc or mac).
% [How I derived this calibration?]
% 1. measure ocean and specbos together
% 2. calculate scalars = radiance/ocean signals
% 3. run cftool; x = (wavelength) y = (scalar=rad/ocean)
% 4. a quadratic model for ocean optics (scalar).
% 5. curve-calibrated radiance = (ocean signals)*(duadratic scalar).
%==============================================================================================%
% function radianceout = ocean2rad(signals,interval)
function radianceout = ocean2rad(signals,interval)
    if (nargin < 2)
        interval = 1;
    end
    signals(:,1) = floor(signals(:,1));
    % regularizing ocean signals
%     regsing = regularize(signals, interval); % wrong result!
    regsing = regularize(signals);
    % build the characterization matrix
%------------------------------------------------------- 
% >> Ver. 4 (qudratic model plus wavelength offset correction)
% ver.4 (20-average measurements of Ocean)
% DATE: 06/23/2011
% without outliers.				
% Linear model Poly2:				
%      f(x) = p1*x^2 + p2*x + p3				
% Coefficients (with 95% confidence bounds):				
% p1	=	4.14E-07	(3.101e-007,	5.175e-007)
% p2	=	-0.0004355	(-0.0005581,	-0.0003129)
% p3	=	0.1179	(0.08221,	0.1535)
% 				
% Goodness	of	fit:		
% 	SSE:	6.34E-05		
% 	R-square:	0.9187		
% 	Adjusted	R-square:	0.9117	
% 	RMSE:	0.00166		
%------------------------------------------------------- 
       p1 = 4.14E-07;%  (3.022e-007, 5.151e-007)
       p2 = -0.0004355;%  (-0.0005554, -0.000304)
       p3 =  0.1179;%  (0.07983, 0.1528)
       
       scalar = 0.001; % from GretagMacbeth EyeOne measurement (x 10^-3) into Radiance[W/(sr*sqm*nm)]
%        [Evaluation] Compared with training Eyeone measurements, 
%        CV error was CV:	22.43%
%------------------------------------------------------- 
    x = regsing(:,1);
   chm_fx = p1.*x.^2 + p2.*x + p3; % for normal (ver.2)
%     chm_fx = p1.*x.^4 + p2.*x.^3 + p3.*x.^2 + p4.*x + p5; % for ver.3 
    radiance = zeros(size(regsing));
    radiance(:,1) = regsing(:,1);
    for i=2:size(regsing,2)
        radiance(:,i) = scalar.*chm_fx.*regsing(:,i);
    end
    radianceout = regularize(radiance,interval);
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
                regsign(k,1) = prewave+1; % (because of flooring) adding one makes it more accurate! (tested with 10nm)
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