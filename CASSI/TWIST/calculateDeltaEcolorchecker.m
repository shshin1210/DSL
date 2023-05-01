clear;

wvls = transpose([450:10:700]);

rad_data = csvread('jetty_radiance.csv');
%recon_data = csvread('synthetic_radiance.csv');
recon_data = csvread('recon_radiance.csv');

illumin = 'd65';

[lambdaCMF, xFcn, yFcn, zFcn] = colorMatchFcn('judd_vos');

wvl_indicies = 71:10:321;
xFcn = xFcn(wvl_indicies);
yFcn = yFcn(wvl_indicies);
zFcn = zFcn(wvl_indicies);

%White_XYZ = [xFcn * whiteref, yFcn * whiteref, zFcn * whiteref];

% Ref_XYZ = Ref2XYZ(wvls, rad_ref, illumin);
% Ref_XYZ = Ref_XYZ./White_XYZ;
% Ref_Lab = xyz2lab(Ref_XYZ);

%{
target_labs = zeros(3,12);

for i=1:16
    rad_target = data(:,i);
    
    target_XYZ = Ref2XYZ(wvls, rad_target, illumin);
    target_XYZ = target_XYZ./White_XYZ;
    target_Lab = xyz2lab(target_XYZ);
    
    target_labs(:,i) = target_Lab;
end

for i=1:3
    ref_lab = transpose(target_labs(:,13 + i));
    for j=1:4
        target_lab = transpose(target_labs(:,(i-1) * 4 + j));
        
        fprintf('%f\n', deltaE2000(ref_lab, target_lab));
    end
end
%}


for i=1:26
    ref = rad_data(i,:);
    ref = ref./ ref(1);
    
    target = recon_data(i,:);
    
    w = transpose(ref) \ transpose(target);
    %w = ref(1) / target(1);
    
    rad_data(i,:) = ref;
    recon_data(i,:) = target .* w;
end

ref = rad_data(:,1);
ref_white = [xFcn * ref, yFcn * ref, zFcn * ref];

target = recon_data(:,1);
target_white = [xFcn * target, yFcn * target, zFcn * target];

deltaEs = [];
for i=1:24
    ref = rad_data(:,i);
    target = recon_data(:,i);
    
    ref_XYZ = [xFcn * ref, yFcn * ref, zFcn * ref];
    ref_XYZ = ref_XYZ./ref_white;
    
    target_XYZ = [xFcn * target, yFcn * target, zFcn * target];
    target_XYZ = target_XYZ./target_white;
    
    ref_lab = xyz2lab(ref_XYZ);
    target_lab = xyz2lab(target_XYZ);
    
    deltaE = deltaE2000(ref_lab, target_lab);
    fprintf('%f\n', deltaE);
    deltaEs(i) = deltaE;
end

deltaEmean = mean(deltaEs(2:end));
stdE = std(deltaEs(2:end));
maxE = max(deltaEs(2:end));
minE = min(deltaEs(2:end));

fprintf('==============\n');
fprintf('mean: %f\n', deltaEmean);
fprintf('std: %f\n', stdE);
fprintf('max: %f\n', maxE);
fprintf('min: %f\n', minE);
fprintf('==============\n');

% deltaE = deltaE2000(Ref_Lab, target_Lab);
% 
% xyz = target_XYZ;
% rgb = xyz2rgb(xyz);
% 
% a = zeros(100,100,3);
% a(:,:,1) = rgb(1);
% a(:,:,2) = rgb(2);
% a(:,:,3) = rgb(3);
% imshow(a);