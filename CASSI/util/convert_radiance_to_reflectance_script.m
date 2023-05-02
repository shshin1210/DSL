
clear
close all;

addpath('mkcassi');

folder_name = uigetdir;
folder_name = [folder_name '\'];

lambdas = 430:10:650;

fn_list = dir([folder_name '*0.png']);

cur_fn = [folder_name fn_list(1).name];
L = size(fn_list,1);
im_cur = imread(cur_fn);
[N,M,~] = size(im_cur);

wvls2b = 430:10:650;

x_recon = zeros(N,M,L);
x_recon(:) = eps;
% wvls2b = zeros(L,1);
for i = 1:numel(fn_list)
    cur_fn = [folder_name fn_list(i).name];
    x_recon(:,:,i) = im2double( imread([cur_fn]) );
end

%%
msimg_reflectance = make_reflectance_using_white_patch(x_recon, lambdas);

output_folder_name = [folder_name 'reflectance'];
mkdir(output_folder_name);
for i = 1:numel(fn_list)
	imwrite(msimg_reflectance(:,:,i), [output_folder_name fn_list(i).name]);
end

%%
params.lambdas = wvls2b;

% stuffed toy parameters
% wpatch: 0.6
% minv: 0.07
% maxv: 0.95

params.min_v = 0.02;%0.07;
params.max_v = 0.95;%0.95;

im_srgb = spec2srgb(msimg_reflectance, params);
figure; imshow(im_srgb);
imwrite(im_srgb, [folder_name 'im_srgb.png']);
