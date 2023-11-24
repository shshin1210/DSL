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
for i=1:(size(A,1))
   Ad(i,:) = A(i+1,:)./WH;
%   Ad(i,:) = A(i,:);
end
A = Ad;

B = subrad(:,(2:end))';
WH = subrad(:,2)';
for i=1:(size(B,1))
   Bd(i,:) = B(i+1,:)./WH;
%   Bd(i,:) = B(i,:);
end
B = Bd;

wvls2b = subrad(:,1);


% per wavelength
for i=1:size(B,2)
    k(i) = curvefitzero(A(:,i), B(:,i));
    B2(:,i) = A(:,i)*k(i);
    cvs(i) = cv(B(:,i),B2(:,i));
    
    % show results
    figure;
    plot([0:0.0001:0.1],[0:0.0001:0.1]); hold on
    scatter(B(:,i), B2(:,i), '*');
    xlabel('Measured radiance');
    ylabel('Estimated radiance');
    title(['Camera raw responsivity (Training): ' num2str(wvls2b(i)) 'nm']);
%    ylim([0,0.1]);
%    xlim([0,0.1]);
    hold off;
    %print('-dpng',['TE_rad-cam_' num2str(floor(wvls2b(i))) 'nm.png']);
    %close;
end

% extrapolate the last value 'nearest' of k
method = 'nearest'; % safer NB the noise of NIR (but it will under-estimate NIR)
wavei = [359;364;369;375;382;389;397;405;415;425;437;450;464;480;497;516;523;530;537;544;552;560;568;577;586;595;604;614;624;635;645;657;668;680;692;705;718;732;746;761;776;791;807;824;841;859;878;897;916;937;958;980;1002;];
ki = interp1(wvls2b,k,wavei,method,'extrap');

% manual tweak % remove the IR peak and unknown IR response
method = 'nearest';
kim = ki(1:46);
kim(44,1) = (ki(43,1)+ki(45,1))/2;% 824nm
waveb = wavei(1:46);
kim_final = interp1(waveb,kim,wavei,method,'extrap');