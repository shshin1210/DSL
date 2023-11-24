% Thi is the without interpolation 
%---------------------------------------------------------


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
WH = cam2(:,2)'; % 1: wavelength, 2: spectralon
for i=1:(size(A,1)-1)
   Ad(i,:) = A(i+1,:)./WH;
end
A = Ad;

B = subrad(:,(2:end))';
WH = subrad(:,2)';
for i=1:(size(B,1)-1)
   Bd(i,:) = B(i+1,:)./WH;
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

% extrapolation methods
% 'nearest' Nearest neighbor interpolation
% 'linear'Linear interpolation (default)
% 'spline'Cubic spline interpolation
% 'pchip'Piecewise cubic Hermite interpolation
% 'cubic'(Same as 'pchip')
% 'v5cubic'Cubic interpolation used in MATLAB 5. This method does not extrapolate. Also, if x is not equally spaced, 'spline' is used.
method = 'nearest'; % most safer
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
[cvd, stdv]=cv(BT,BT2) % CV error: 13.6111% % after normalization
cal_test = BT2';


%-------------------------------------------------------
% Performance without white balancing
% cvd = 5.8598
% stdv = 0.2475
% cvd = 13.1906
% stdv = 0.2639