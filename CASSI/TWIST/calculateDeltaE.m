wvls = transpose([450:10:700]);

data = csvread('video_spectrums.csv');

whiteref = [1.0015
1.0015
1.0012
1.0012
1.0006
1.0013
1.0012
1.001
1.0013
1.0009
1.0012
1.0011
1.0008
1.001
1.0009
1.0005
1.0004
1.0005
1.0007
1.0004
1.0004
1.0005
1.0005
1.0005
1.0005
];

illumin = 'd50';

White_XYZ = Ref2XYZ(wvls, whiteref, illumin);

% Ref_XYZ = Ref2XYZ(wvls, rad_ref, illumin);
% Ref_XYZ = Ref_XYZ./White_XYZ;
% Ref_Lab = xyz2lab(Ref_XYZ);

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