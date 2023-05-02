function cassipack( result_path )

%% Read RGB
im_rgb = imread([result_path '/rgb.png']);

[h, w, ~] = size(im_rgb);

pad = 8;

cell_w = floor((w - pad*4) / 5);
cell_h = floor((h - pad*4) / 5);

%% Read x_recon
matfiles = dir([result_path '/*.mat']);

load([result_path '/' matfiles(1).name]);

mkdir([result_path '/linear']);
mkdir([result_path '/gamma']);

num_wvls = size(wvls2b, 2);

imbands = zeros(h, w, 3, num_wvls);

for i = 1:num_wvls
    wvl = wvls2b(i);
    rgb = spectrumRGB(wvl);
    imband(:,:,1) = x_recon(:,:,i) * num_wvls * rgb(1);
    imband(:,:,2) = x_recon(:,:,i) * num_wvls * rgb(2);
    imband(:,:,3) = x_recon(:,:,i) * num_wvls * rgb(3);
    
    imbands(:,:,:,i) = imband;
end

for i = 1:num_wvls
    wvl = wvls2b(i);
    imband = imbands(:,:,:,i);

    imwrite(imband, sprintf('%s/%s/%dnm.png', result_path, 'linear', wvl));

    imband_gamma = imband .^ (1/2.2);

    imwrite(imband_gamma, sprintf('%s/%s/%dnm.png', result_path, 'gamma', wvl));
end
% 
% for i = 4:28
%     imband = imbands(:,:,:,i);
%     imband_gamma = real(imband .^ (1/2.2));
%     
%     imband_resized = imresize(imband_gamma, [cell_h cell_w]);
%     
%     x = (mod(i-4,5) * (cell_w + pad)) + 1;
%     y = (floor((i-4)/5) * (cell_h + pad)) + 1;
% end

end

