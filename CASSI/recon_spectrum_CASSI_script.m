clear all;
close all;

%% parameters
data_path = 'Y:\papers\prism\scene\20170215_spectrum_recon_test\synthetic_columbia_glass_tiles_realspec2rgb\';
result_path = [data_path 'result_CASSI_1\'];
if ~exist(result_path, 'dir')
    mkdir(result_path);
end
ms_path = [data_path 'msimgs\'];
% captured_fn = 'PMVIS_input.png';
% opt.lambda_min = 400; opt.lambda_max = 700; opt.n_lambda = 31;
opt.lambda_min = 430; opt.lambda_max = 650; opt.n_lambda = 23;
lambdas = opt.lambda_min:10:opt.lambda_max;
% spec2grey = ones(1, opt.n_lambda)/opt.n_lambda;    %'spec2rgb.mat';
spec2grey = ones(1, opt.n_lambda);    %'spec2rgb.mat';
table_fn = 'dispersion_model.mat';
spec2rgb_fn = 'spec2rgb.mat';
addpath('util\');
addpath('recon\');
addpath('TWIST\');
addpath('model\');
global spec_plot_regions;
path_parts = strsplit(data_path, filesep);
spec_plot_regions = designate_spec_region( path_parts{end-1} );

% read the spec2rgb matrix
load([data_path spec2rgb_fn]);

%% load mapping table
load([data_path table_fn]);

%% read ground truth multispectral images
msim_list = dir([ms_path '*.png']);
nChannels = numel(msim_list);

im0 = imread([ms_path msim_list(1).name]);
[N,M] = size(im0);
% prepare multi-spectral images
msImg = [];
for i = 1:nChannels
    im = im2double(imread([ms_path msim_list(i).name]));
    msImg = cat(3, msImg, im);
end
% msImg = msImg(:,:,(nChannels-opt.n_lambda)+1:end);
display_spectral_images(msImg);

params.lambda_roi_min_ind = 2;
params.lambda_roi_max_ind = 24;
params.lambda_n_roi = opt.n_lambda;
params.roi_max_r = N;   params.roi_min_r = 1; params.roi_max_c = M;     params.roi_min_c = 1; 
msImg = msImg(params.roi_min_r:params.roi_max_r, params.roi_min_c:params.roi_max_c, params.lambda_roi_min_ind:params.lambda_roi_max_ind);
p_ref_lambda2lambda = model_crop_phi_with_rect(p_ref_lambda2lambda, params);
lambdas = lambdas(params.lambda_roi_min_ind:params.lambda_roi_max_ind);


%% Crop coded mask
nChannels = opt.n_lambda;
coded_mask = double(rand(N,M) >= 0.5);
% coded_mask = im2double(imread('560nm.png'));
% [h_cm, w_cm] = size(coded_mask);
% h_cm_ctr = round(h_cm/2); w_cm_ctr = round(w_cm/2);
% coded_mask = coded_mask((h_cm_ctr-round(N/2)):(h_cm_ctr+round(N/2)-1),(w_cm_ctr-round(M/2)):(w_cm_ctr+round(M/2)-1));
coded_mask = repmat(coded_mask, [1,1,nChannels]);
save([result_path 'coded_mask.mat'], 'coded_mask');

%% make the captured grayscale image
im_observed = compute_A_mul_any_coded_mask(msImg, p_ref_lambda2lambda, spec2grey, 1, coded_mask);

figure;
imshow(im_observed./max(im_observed(:))); title('observed');
imwrite(im2uint16(im_observed./max(im_observed(:))), [data_path 'CASSI_input.png']);

%% compute the pad for psnr
global psnr_pad;

% psnr_pad= ;

p_self = p_ref_lambda2lambda(:,:,:,end);
p_spatial = squeeze(p_ref_lambda2lambda(:,:,:,1));
diff_c = p_self(:,:,1) - p_spatial(:,:,1);
diff_r = p_self(:,:,2) - p_spatial(:,:,2);

% discard the pixels which have no corresponding points on the
% image
diff_r = diff_r(:,210:end);
diff_c = diff_c(:,210:end);

[dispersion_mag, dispersion_angle] = imgradient(diff_c, diff_r);
psnr_pad = round(max(dispersion_mag(:))+1);


%% reconstruct spectrum
% parameter setting for TWIST
params.tau = 3e-1; %0.15; % smoothing factor parameters
params.tv_iter = 20; % numger of iteration in a single denoise (for tvdenoise.m)
params.iterA = 40; % max iteration for TwIST
params.tolA = 1e-5; % Iteration stop criteria in TwIST
save([result_path 'params.mat'], 'params', 'spec2grey');

[im_spec, obj_twist] = cassirecon_coded_mask_sh(im_observed, p_ref_lambda2lambda, spec2grey, coded_mask, params);
im_spec = double(im_spec);

% save the results
display_spectral_images_psnr(double(im_spec), msImg, lambdas);
save_figure_to_pdf(gcf, [result_path 'psnr.pdf']);
% print([result_path 'psnr.pdf'], '-fillpage', '-dpdf');
for i = 1:size(msImg,3)
    imwrite(im_spec(:,:,i), [result_path int2str(lambdas(i)) '.png']);
end

% visualize and save the results
p_ref_lambda2_ref_lambda = zeros(size(p_ref_lambda2lambda));
[col_cand, row_cand] = meshgrid(1:M, 1:N);
p_ref_lambda2_ref_lambda(:,:,1,:) = repmat(col_cand, 1, 1, nChannels);
p_ref_lambda2_ref_lambda(:,:,2,:) = repmat(row_cand, 1, 1, nChannels );
spec2rgb = spec2rgb(:,params.lambda_roi_min_ind:params.lambda_roi_max_ind);
im_recovered_rgb = compute_A_mul_any_v3(im_spec, p_ref_lambda2_ref_lambda, spec2rgb, 1);
imwrite(im_recovered_rgb, [result_path 'recovered_RGB.png']);

figure; plot(obj_twist, '--bo'); 
title('pcg obj. value');
xlabel('iterations');
ylabel('res norm');
save_figure_to_pdf(gcf, [result_path 'objective_residual.pdf']);

% colorchecker psnr
% spectrum_comparison_plot2(im_spec, msImg, lambdas);
spectrum_comparison_plot_given_region(im_spec, msImg, lambdas);

save_figure_to_pdf(gcf, [result_path 'rmse.pdf'], 9);

% spectral images
display_spectral_images_psnr(im_spec, msImg, lambdas);
save_figure_to_pdf(gcf, [result_path 'psnr_final.pdf']);

[im_spec_color_coded,im_spec_color_coded_concat] = apply_color_coding_to_hsimgs(im_spec, lambdas);
for i = 1:nChannels
    imwrite(im_spec_color_coded(:,:,:,i), [result_path 'color_' int2str(lambdas(i)) '.png']);
end
imwrite(im_spec_color_coded_concat, [result_path 'color_concat.png']);

save([result_path 'recon.mat'], 'im_spec');
close all;