% This is the interpolated version not good. 
%----------------------------------------------------------
% apply cosine raw for the 1/45 measurements
wav = [179:1005]';
rad2 = [rad_training(:,1), rad_training(:,2:end).*(1/cos(pi/4))];

% Radiance measurements are limited. 
% so we extrapolate the radiance measurements.

% extrapolation methods
% 'nearest' Nearest neighbor interpolation
% 'linear'Linear interpolation (default)
% 'spline'Cubic spline interpolation
% 'pchip'Piecewise cubic Hermite interpolation
% 'cubic'(Same as 'pchip')
% 'v5cubic'Cubic interpolation used in MATLAB 5. This method does not extrapolate. Also, if x is not equally spaced, 'spline' is used.
method = 'nearest'; % most safer

% row-direction interpolation (this should be done first)
radi(:,1) = wav(:,1);
for i=2:size(rad_training,2)
    xi = wav(:,1);
    x = rad_training(:,1);
    Y = rad2(:,i);
    yi = interp1(x,Y,xi,method,'extrap');
    radi(:,i) = yi;
end


cam2 = cam_training;

% sampling radiance
k=1;
for i=1:size(radi,1)
    for j=1:size(cam2,1)
        if (cam2(j,1) == radi(i,1))
            subrad(k,:) = radi(i,:);
            k = k + 1;
        end
    end
end

% Normalize white spectrum
% this includes wavelength and spectralon
A = cam2(:,(3:end))'; 
WH = cam2(:,2)'; % 1: wavelength, 2: spectralon
for i=1:(size(A,1))
   Ad(i,:) = A(i,:)./WH;
end
A = Ad;

B = subrad(:,(3:end))';
WH = subrad(:,2)';
for i=1:(size(B,1))
   Bd(i,:) = B(i,:)./WH;
end
B = Bd;

%X = A\B; %QR decomposition. Don's use this 'inverse operator' -> Ended up with zero-filled matrix
%X = inv(A'*A)*A'*B;% not working -> Failure
X = pinv(A)*B; % this is better than QR;

B2 = A*X;
B2(B2<0) = 0; % remove negative radiance
[cvd, stdv] = cv(B,B2) %  5.8598% error in training (with white balancing)

% extrapolate the calibration matrix
% total is 53 but we have 47 only.


%--------------------------------------------------------------------%
% test evaluation
rad2t = [rad_test(:,1), rad_test(:,2:end).*(1/cos(pi/4))];
radit(:,1) = wav(:,1);
for i=2:size(rad_test,2)
    xi = wav(:,1);
    x = rad_test(:,1);
    Y = rad2t(:,i);
    yi = interp1(x,Y,xi,method,'extrap');
    radit(:,i) = yi;
end
% subsampling 
k=1;
for i=1:size(radit,1)
    for j=1:size(cam_test,1)
        if (cam_test(j,1) == radit(i,1))
            subradt(k,:) = radit(i,:);
            k = k + 1;
        end
    end
end


AT = cam_test(:,(3:end))';
WH = cam_test(:,2)';
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
[cvd, stdv]=cv(BT,BT2) % CV error: 13.6111% % after normalization
cal_test = BT2';

%-------------------------------------------------------
% Performance without white balancing
% cvd = 5.8598
% stdv = 0.2475
% cvd = 13.1906
% stdv = 0.2639