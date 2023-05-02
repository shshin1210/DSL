addpath

folder_name = uigetdir;
folder_name = [folder_name '\'];

input_fn = 'input.png';
im_in = imread([folder_name input_fn]);
output_fn = 'recovered.png';
im_rec = imread([folder_name output_fn]);

figure; 
imshow(im_in); title('input');

figure; 
imshow(im_rec); title('output');

fn_list = dir([folder_name '*0.png']);

cur_fn = [folder_name fn_list(1).name];
L = size(fn_list,1);
im_cur = imread([cur_fn]);
[N,M,~] = size(im_cur);

% wvls2b = 420:10:700;

x_recon = zeros(N,M,L);
x_recon(:) = eps;
wvls2b = zeros(L,1);
for i = 1:numel(fn_list)
    cur_fn = [folder_name fn_list(i).name];
    wvls2b(i) = str2double( fn_list(i).name(1:3) );
    x_recon(:,:,find(wvls2b == wvls2b(i))) = im2double( imread([cur_fn]) );
end
x_recon = single(x_recon);

save([folder_name 'result.mat'], 'wvls2b', 'x_recon');

% 
% % sRGB
% spec2rgb;
% im_RGB = compute_A_mul_any_v3(x_recon, index_before_warp, spec2rgb);
% 
% % apply 
% [im_RGB_WB, wb_factors] = apply_WB_and_GAMMA(im_RGB, 2.2);
% 
% im_sRGB;
% 
