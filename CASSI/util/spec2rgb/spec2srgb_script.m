clear
close all;

% addpath('mkcassi');

folder_name = uigetdir;
folder_name = [folder_name '\'];

% input_fn = 'input.png';
% im_in = imread([folder_name input_fn]);
% output_fn = 'recovered.png';
% im_rec = imread([folder_name output_fn]);
% 
% figure; 
% imshow(im_in); title('input');
% 
% figure; 
% imshow(im_rec); title('output');

fn_list = dir([folder_name '*0.png']);
% fn_list = fn_list(1:23);

cur_fn = [folder_name fn_list(1).name];
L = size(fn_list,1);
im_cur = imread([cur_fn]);
[N,M,~] = size(im_cur);

wvls2b = 430:10:650;

x_recon = zeros(N,M,L);
x_recon(:) = eps;
% wvls2b = zeros(L,1);
for i = 1:numel(fn_list)
    cur_fn = [folder_name fn_list(i).name];
%     wvls2b(i) = str2double( fn_list(i).name(1:3) );
%     wvls2b(i) = str2double( fn_list(i).name(4:6) );
%     wvls2b(i) = str2double( fn_list(i).name(9:11) );
    x_recon(:,:,i) = im2double( imread([cur_fn]) );
end
x_recon = single(x_recon);

params.lambdas = wvls2b;
% addpath('..\util\');

%%
params.min_v = 0.06;%0.07;
params.max_v = 1;%0.95;

im_srgb = spec2srgb(x_recon, params);
figure; imshow(im_srgb);
imwrite(im_srgb, [folder_name 'im_srgb_radiance.png']);



% 
% save([folder_name 'result.mat'], 'wvls2b', 'x_recon');
% [lambda_illum, illum] = illuminantFcn('d65');
% lambda_idx = find(lambda_illum==wvls2b(1)):10:find(lambda_illum==wvls2b(end));
% illum = illum(lambda_idx);
% 
% x_radiance = x_recon;
% for i = 1:L
%     x_radiance(:,:,i) = x_radiance(:,:,i) * illum(i)/100;
% end
% 
% [lambda_cmf, xfcn, yfcn, zfcn] = colorMatchFcn('1931_full');
% lmbcmf_idx = find(lambda_cmf==wvls2b(1)):10:find(lambda_cmf==wvls2b(end));
% xfcn = repmat(reshape(xfcn(lmbcmf_idx),1,1,[]),N,M);    yfcn = repmat(reshape(yfcn(lmbcmf_idx),1,1,[]),N,M);     zfcn = repmat(reshape(zfcn(lmbcmf_idx),1,1,[]),N,M); 
% 
% x_xyz = zeros(N,M,3);
% x_xyz(:,:,1) = sum(x_radiance.*xfcn,3);
% x_xyz(:,:,2) = sum(x_radiance.*yfcn,3);
% x_xyz(:,:,3) = sum(x_radiance.*zfcn,3);
% 
% y_denom = max(reshape(x_xyz(:,:,2),[],1));
% x_xyz = 1.0 .* x_xyz ./ y_denom;
% 
% x_srgb = XYZ2sRGB(x_xyz);
% 
% figure; imshow(x_srgb)

% sRGB
% spec2rgb;
% im_sRGB = compute_A_mul_any_v3(x_recon, index_before_warp, spec2rgb);

% apply_homography
% im_sRGB;

