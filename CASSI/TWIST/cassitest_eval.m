

paths = {'single', 'single_full', '5views', '5views_flipped', '9views', '9views_flipped'};

load('balloons_ms/balloons_gt.mat');

for path_cell = paths
    path = path_cell{1};
    
    fullpath = ['balloons_ms/' path '/result/'];
    
    matfiles = dir([fullpath '*.mat']);
    
    load([fullpath matfiles(1).name]);
    
    x_recon = imresize(im2double(x_recon), [512, 512]) * 26;

    %% PSNR
    MAX = max(Cu(:));
    SE = (Cu - x_recon) .^2;
    MSE = sum(SE(:)) / numel(Cu);   
    PSNR = 20 * log10(MAX / sqrt(MSE));
    
    fprintf('PSNR: %f\n', PSNR);
    fprintf('SSIM: %f\n', ssim(Cu, x_recon));
end