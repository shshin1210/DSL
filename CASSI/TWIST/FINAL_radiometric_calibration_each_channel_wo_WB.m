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

% Thi is the without interpolation 
%---------------------------------------------------------
% Stage 1/2: Linearization
%---------------------------------------------------------
disp 'Stage1/2: Linearization'
% apply cosine raw for the 1/45 measurements
rad2 = [rad_training(:,1), rad_training(:,2:end).*(1/cos(pi/4))];
cam2 = cam_training(1:47,:);

% sampling radiance
k=1;
for i=1:size(rad2,1)
    for j=1:size(cam2,1)
        if (cam2(j,1) == rad2(i,1))
            subrad(k,:) = rad2(i,:);
            k = k + 1;
        end
    end
end

% Normalize white spectrum
% this includes wavelength and spectralon
A = cam2(:,(2:end))'; 
%WH = cam2(:,2)'; % 1: wavelength, 2: spectralon

B = subrad(:,(2:end))';
%WH = subrad(:,2)';

wvls2b = subrad(:,1);


% per wavelength
for i=1:size(B,2)
    k(i) = curvefitzero(A(:,i), B(:,i));
    B2(:,i) = A(:,i)*k(i);
%    cvs(i) = cv(B(:,i),B2(:,i));
    
    % show results
%     figure;
%     plot([0:0.0001:0.1],[0:0.0001:0.1]); hold on
%     scatter(B(:,i), B2(:,i), '*');
%     xlabel('Measured radiance');
%     ylabel('Estimated radiance');
%     title(['Camera raw responsivity (Training): ' num2str(wvls2b(i)) 'nm']);
%     ylim([0,0.1]);
%     xlim([0,0.1]);
%     hold off;
%     print('-dpng',['Calib_' num2str(floor(wvls2b(i))) 'nm.png']);
%     close;
end

%close all;

% extrapolate the last value 'nearest' of k
method = 'nearest'; % safer NB the noise of NIR (but it will under-estimate NIR)
wavei = [359;364;369;375;382;389;397;405;415;425;437;450;464;480;497;516;523;530;537;544;552;560;568;577;586;595;604;614;624;635;645;657;668;680;692;705;718;732;746;761;776;791;807;824;841;859;878;897;916;937;958;980;1002;];
ki = interp1(wvls2b,k,wavei,method,'extrap');
K_FINAL = ki;

ki = ki(1:size(A,2),1);

kk = repmat(ki',[size(A,1), 1]);
% apply calibration (this is the hadama product)
B2 = A.*kk;

disp 'Characterization result'

% NORMALIZATION IS NECESSARY -> AS WE USE THIS SYSTEM FOR REFLECTANCE
% MEASUREMENTS
RW1 = B(1,:);
RW1 = repmat(RW1,[size(B,1), 1]);
B = B./RW1;

RW2 = B2(1,:);
RW2 = repmat(RW2,[size(B2,1), 1]);
B2 = B2./RW2;

% CALCULATE REFLECTANCE ACCURACY
cc1 = corrcoef(B(:),B2(:));
r21 = cc1(1,2)^2 % RSQ 0.9867
[meddiff,stdv] = medianreldiff(B,B2)
cv1 = cv(B(:),B2(:)) %  17.1438


%--------------------------------------------------------------------%
% test evaluation
%--------------------------------------------------------------------%
rad2t = [rad_test(:,1), rad_test(:,2:end).*(1/cos(pi/4))];
% subsampling 
k=1;
for i=1:size(rad2t,1)
    for j=1:size(cam_test,1)
        if (cam_test(j,1) == rad2t(i,1))
            subradt(k,:) = rad2t(i,:);
            subcamt(k,:) = cam_test(j,:);
            k = k + 1;
        end
    end
end


AT = subcamt(:,(2:end))';
BT = subradt(:,(2:end))';

ki = ki(1:size(AT,2),1);
kk = repmat(ki',[size(AT,1), 1]);
% apply calibration (this is the hadama product)
BT2 = AT.*kk;
%BT2 = AT*X;

BT2(BT2<0) = 0;

disp 'After applying Characterization on test'

RWT1 = BT(1,:);
RWT1 = repmat(RWT1,[size(BT,1), 1]);
BT = BT./RWT1;

RWT2 = BT2(1,:);
RWT2 = repmat(RWT2,[size(BT2,1), 1]);
BT2 = BT2./RWT2;

cc1 = corrcoef(BT2(:),BT(:));
r21 = cc1(1,2)^2 % RSQ 0.9867
[meddiff,stdv] = medianreldiff(BT,BT2)
cv1 = cv(BT2(:),BT(:)) %  17.1438




%{  
RESULTS FOR ABS RADIANCE
-----------------------------------------------
Characterization result
<TRAINING>  --> REPORT
RSQ = 0.9867
meddiff = 12.1473
stdv = 0.0215
cv1 = 17.1438

<TEST> --> NOT MEANINGFUL DUE TO EXPOSURE SETTING CHANGE
RSQ = 0.9879
meddiff = 38.2346
stdv = 0.0233
cv1 = 47.5081
%}
%{  
RESULTS FOR REFLECTANCE
-----------------------------------------------
Characterization result
<TRAINING> --> NOT MEANINGFUL
RSQ = 0.9508
meddiff = 11.9792
stdv = 0.2561
cv1 = 21.4338

<TEST> ---> REPORT
RSQ = 0.9758
meddiff = 8.4450
stdv = 0.2960
cv1 = 13.4018
%}




%{

%----------------------------------------------------------------
% This is totally wrong
%----------------------------------------------------------------
% manual tweak % remove the IR peak and unknown IR response
% method = 'nearest';
% kim = ki(1:46);
% kim(44,1) = (ki(43,1)+ki(45,1))/2;% 824nm -> This is totally wrong
% waveb = wavei(1:46);
% kim_final = interp1(waveb,kim,wavei,method,'extrap');
% 
% kim_final = kim_final(1:47,1);
% 
% kim_final = repmat(kim_final',[241, 1]);
% % apply calibration (this is the hadama product)
% B22 = A.*kim_final;
% 
% cc2 = corrcoef(B(:),B22(:));
% r22 = cc2(1,2)^2 % RSQ 0.9867
% cv2 = cv(B(:),B22(:)) %  17.1438

%----------------------------------------------------------------
% Second stage: White balance -> reflectance matrix
%----------------------------------------------------------------
disp 'Stage2/2: Characterization'

% apply cosine raw for the 1/45 measurements
rad2 = [rad_training(:,1), rad_training(:,2:end).*(1/cos(pi/4))];
cam2 = cam_training(1:47,:);


% sampling radiance
k=1;
for i=1:size(rad2,1)
    for j=1:size(cam2,1)
        if (cam2(j,1) == rad2(i,1))
            subrad(k,:) = rad2(i,:);
            k = k + 1;
        end
    end
end

%------------------------------------------------
% applying linearization matrix
A = cam2(:,2:end)'; 
ki = ki(1:size(A,2),1);
kk = repmat(ki',[size(A,1), 1]);
% apply calibration (this is the hadama product)
A2 = A.*kk;

disp 'After applying Linearization'
cc1 = corrcoef(B(:),A2(:));
r21 = cc1(1,2)^2 % RSQ 0.9867
cv1 = cv(B(:),A2(:)) %  17.1438

%------------------------------------------------
% Normalize white spectrum -> reflectance calculation
% this includes wavelength and spectralon
%A = cam2'; 
WH = repmat(A2(1,:), [size(A2,1),1]);
WHr = repmat(B(1,:), [size(B,1),1]);
% A2 = A2./WH;
% B = B./WHr;

X = pinv(A2)*B; % this is better than QR;

B2 = A2*X;
B2(B2<0) = 0; % remove negative radiance


disp 'After applying Characterization on training'
cc1 = corrcoef(B(:),B2(:));
r21 = cc1(1,2)^2 % RSQ 0.9867
cv1 = cv(B(:),B2(:)) %  17.1438

% extrapolate the calibration matrix
% total is 53 but we have 47 only.

% extrapolation methods
% 'nearest' Nearest neighbor interpolation
% 'linear'Linear interpolation (default)
% 'spline'Cubic spline interpolation
% 'pchip'Piecewise cubic Hermite interpolation
% 'cubic'(Same as 'pchip')
% 'v5cubic'Cubic interpolation used in MATLAB 5. This method does not extrapolate. Also, if x is not equally spaced, 'spline' is used.
method = 'linear'; % most safer

% row-direction interpolation (this should be done first)
for i=1:47
    xi = cam_training(:,1);
    x = cam2(:,1);
    Y = X(:,i);
    yi = interp1(x,Y,xi,method,'extrap');
    Xi(:,i) = yi;
end

% column-direction interpolation (this should be done second)
for i=1:53
    xi = cam_training(:,1);
    x = cam2(:,1);
    Y = Xi(i,:);
    yi = interp1(x,Y,xi,method,'extrap');
    Xii(i,:) = yi;
end

Xii = zeros(53,53);
Xii(1:47,1:47) = X;

%--------------------------------------------------------------------%
% test evaluation
%--------------------------------------------------------------------%
rad2t = [rad_test(:,1), rad_test(:,2:end).*(1/cos(pi/4))];
% subsampling 
k=1;
for i=1:size(rad2t,1)
    for j=1:size(cam_test,1)
        if (cam_test(j,1) == rad2t(i,1))
            subradt(k,:) = rad2t(i,:);
            subcamt(k,:) = cam_test(j,:);
            k = k + 1;
        end
    end
end


AT = subcamt(:,(3:end))';
WH = subcamt(:,2)';
for i=1:(size(AT,1))
   ATd(i,:) = AT(i,:)./WH;
end
AT = ATd;

BT = subradt(:,(3:end))';
WH = subradt(:,2)';
for i=1:(size(BT,1))
   BTd(i,:) = BT(i,:)./WH;
end
BT = BTd;


BT2 = AT*X;
BT2(BT2<0) = 0;

disp 'After applying Characterization on test'
cc1 = corrcoef(BT2(:),BT(:));
r21 = cc1(1,2)^2 % RSQ 0.9867
cv1 = cv(BT2(:),BT(:)) %  17.1438

cal_test = BT2';


%-------------------------------------------------------
% After applying Linearization
% 
% r21 =
% 
%     0.9867
% 
% 
% cv1 =
% 
%    17.1438
% 
% After applying Characterization on training
% 
% r21 =
% 
%     0.9952
% 
% 
% cv1 =
% 
%     6.3203
% 
% After applying Characterization on test
% 
% r21 =
% 
%     0.9803
% 
% 
% cv1 =
% 
%    11.8966

%}