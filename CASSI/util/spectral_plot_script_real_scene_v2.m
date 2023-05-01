% Updated!
% PSNR, SSIM values with the corresponding hyperspectral data for a sample
% patch

close all;
clear all;

% data_path = 'Y:\papers\prism\scene\20170215_spectrum_recon_test\synthetic_columbia_stuffed_toys_with_real_spec2rgb\comp_system_fig\';
% data_path = 'Y:\papers\prism\scene\20170215_spectrum_recon_test\synthetic_vclab_colorchecker_downsampled2\comp_algorithm_fig\';
% data_path = 'Y:\papers\prism\scene\20170215_spectrum_recon_test\synthetic_columbia_feathers_real_spec2rgb\comp_algorithm\';
% data_path = 'Y:\papers\prism\scene\20170215_spectrum_recon_test\synthetic_columbia_beers_real_spec2rgb\comp_algorithm\';
% data_path = 'Y:\papers\prism\scene\20170510_indoor_bottle_doorlock2\spatial_registration_fig\';
% data_path = 'Y:\papers\prism\scene\20170512_daylight_drink_with_gt\result_ic3\5_refine\';
% data_path = 'Y:\papers\prism\scene\20170511_daylight_toys1_with_gt\result_sh_20170515\5_refine\';
% data_path = 'Y:\papers\prism\final_result_figures\real_scenes\siggraph_tungsten\5_refine\';
% data_path = 'Y:\papers\prism\final_result_figures\real_scenes\siggraph_zenon\5_refine\';
% data_path = 'Y:\papers\prism\scene\20170519_darkroom_toys_with_gt\result_ic_ratio4_420680_2\5_refine\';
% data_path = 'Y:\papers\prism\final_result_figures\real_scenes\drink\result_430_680\5_refine\';
% data_path = 'Y:\papers\prism\scene\20170519_tulip2_with_gt\20170519_result_sh\5_refine\';
% data_path = 'Y:\papers\prism\scene\20170517_sign_with_gt\result_ic_ratio4_420680_1\5_refine\';
% data_path = 'Y:\papers\prism\scene\20170519_tulip2_with_gt\20170519_result_sh\refine_corrected_pd\';
% data_path = 'Y:\papers\prism\scene\20170519_tulip2_with_gt\20170519_result_sh\refine_more_spatial_details\refine\';
% data_path = 'Y:\papers\prism\scene\20170522_pot_with_colorchecker_with_gt\20170522_result_sh\5_refine\';
% data_path = 'Y:\papers\prism\scene\20170522_robby_pot_with_gt\20170522_result_sh\tmp\';
data_path = 'Y:\papers\prism\scene\20170522_goose\20170522_result_sh_with_many_edges\gamma3\';
camera_path = 'Y:\papers\prism\calib\20170522_calib\camera\';
load([camera_path 'cam_calib.mat']);
model_path = 'Y:\papers\prism\calib\20170522_calib\model\';
unwarp_model_fn = 'p_ref.mat';
% unwarp_model_fn = 'p_ref_z780_with_synthetic_prism_normal.mat';

% load camera parameters
cam.f = mean(cameraParams.FocalLength);
cam.center_col = cameraParams.PrincipalPoint(1);
cam.center_row = cameraParams.PrincipalPoint(2);

% recon_srgb = imread([data_path 'im_srgb.png']);

% num_roi = input('# of regions of interest: ');
lambdas = 420:10:680;
nCh = numel(lambdas);


%% build the unwarping table
fprintf('building the unwarping table\n');
% [cam, prism, cboard, opt] = init_param();
load([model_path unwarp_model_fn], 'p_ref');
figure; subplot(1,2,1); imagesc(p_ref(:,:,1)); colorbar; subplot(1,2,2); imagesc(p_ref(:,:,2)); colorbar;

[n_pd_y, n_pd_x, ~] = size(p_ref);
p_ref(:,:,1) = round(p_ref(:,:,1) + cam.center_col);
p_ref(:,:,2) = round(p_ref(:,:,2) + cam.center_row);
figure; subplot(1,2,1); imagesc(p_ref(:,:,1)); colorbar; subplot(1,2,2); imagesc(p_ref(:,:,2)); colorbar;

%%
% apply cropping and downsampling
downsampling_ratio = 4;
%min_r = 490; max_r = 3300; min_c = 552; max_c = 5790;
% min_r = 490; max_r = 3300; min_c = 552; max_c = 5790;
% min_r = 1245 - 490; max_r = 3009 - 490; min_c = 1663 - 552; max_c = 5110 - 552;
% min_r = 1245; max_r = 3009; min_c = 1663; max_c = 5110;
first_crop_min_r = 600; first_crop_min_c = 700; first_crop_max_r = 3200; first_crop_max_c = 5790; 
roi_min_r = 900 - first_crop_min_r;    roi_max_r = 3200 - first_crop_min_r;    roi_min_c = 1663 - first_crop_min_c;  roi_max_c = 5110 - first_crop_min_c;

p_ref_d(:,:,1) = (min(p_ref(:,:,1) - first_crop_min_c, first_crop_max_c-first_crop_min_c+1) / downsampling_ratio) - (roi_min_c / downsampling_ratio);
p_ref_d(:,:,2) = (min(p_ref(:,:,2) - first_crop_min_r, first_crop_max_r-first_crop_min_r+1) / downsampling_ratio) - (roi_min_r / downsampling_ratio);


% min_r = 600; max_r = 3200; min_c = 700; max_c = 5790;
% % min_r = 1663 ; max_r = 5110; min_c = 900; max_c = 3300;
% % min_r = 900; max_r = 4000; min_c = 1663; max_c = 5110;
% 
% p_ref_d(:,:,1) = min(p_ref(:,:,1) - min_c, max_c-min_c+1) / downsampling_ratio;
% p_ref_d(:,:,2) = min(p_ref(:,:,2) - min_r, max_r-min_r+1) / downsampling_ratio;
figure; subplot(1,2,1); imagesc(p_ref_d(:,:,1)); colorbar; subplot(1,2,2); imagesc(p_ref_d(:,:,2)); colorbar;


% Read spectral image sets
fprintf('\tReading spectral image result sets\n');

msImg = [];
data_list = dir([data_path '*0.png']);
for i = 1:nCh
    msImg = cat(3, msImg, im2double(imread([data_path data_list(i).name])));
end
[N,M,~] = size(msImg);

% radiance to reflectance

use_reflectance = 0;
if use_reflectance
    fprintf('Use reflectance\n');
    % Use reflectance !!!!!!!!!!!!!
    msimg = make_reflectance_using_white_patch(msImg, lambdas);
    
    for i = 1:nCh
        imwrite(msimg_reflectance(:,:,i), [data_path 'reflectance' data_list(i).name]);
    end
else
    fprintf('Use radiance\n');
end


%%
% if 1
% tmp = convert_pref_to_pd(u_pot, p_ref);
msImg_pd = convert_pref_to_pd(msImg, p_ref_d);
figure; imshow(msImg_pd(:,:,1));
% input('');
%msImg_pd = msImg_pd(1690:4200, 5650:10050, :);
%msImg_pd_crop = msImg_pd(770:2780, 6500:10050, :); 

%%
crop_height = 2300;%2300;
crop_width = 3800;%4000;%4100;
crop_r_min = 1900;
crop_c_min = 6500;%6500;%6000;

% crop_height = 2356;%2300;
% crop_width = 4200;%4100;
% crop_r_min = 1700; 1740;%1800;
% crop_c_min = 4500;

%msImg_pd_crop = msImg_pd(770:2780, 6200:9750, :); 
% msImg_pd_crop = msImg_pd(600:2780, 6200:9750, :); 
% msImg_pd_crop = msImg_pd(550:3100, 4400:8500, :); 
% msImg_pd_crop = msImg_pd(1600:3900, 4700:8400, :); % toy_new
% msImg_pd_crop = msImg_pd(1800:4000, 4300:8800, :); % toy 1

msImg_pd_crop = msImg_pd(crop_r_min:(crop_r_min+crop_height), crop_c_min:(crop_c_min+crop_width), :); % toy 1


msImg_pd_crop = imresize(msImg_pd_crop, 1/downsampling_ratio, 'bicubic');
% else
%     msImg_pd = convert_pref_to_pd(im2double(recovered_RGB_iter6), p_ref_d);
    
% end
if 0
    msImg_pd_crop = imrotate(msImg_pd_crop,-1.5, 'bicubic', 'crop');
    msImg_pd_crop = msImg_pd_crop(20:end-20,:,:);
end

[dN, dM, ~] = size(msImg_pd_crop);
for i = 1:nCh
    figure(3000); imshow(msImg_pd_crop(:,:,i)); title(lambdas(i));
    imwrite(msImg_pd_crop(:,:,i), [data_path 'pd_' data_list(i).name] );
end



%% make color-coded images
fprintf('applying color coding to the images\n');
% msImg_pd_crop = msImg_pd_crop();
msImg_pd_crop_for_color = msImg_pd_crop(:,:,2:24);
lambdas_color = lambdas(2:24);
[~,im_spec_color_coded_concat] = apply_color_coding_to_hsimgs(msImg_pd_crop_for_color, lambdas_color);
[im_spec_color_coded,~] = apply_color_coding_to_hsimgs(msImg_pd_crop, lambdas);
for i = 1:nCh
    imwrite(im_spec_color_coded(:,:,:,i), [data_path 'pd_color' data_list(i).name]);
end
imwrite(im_spec_color_coded_concat, [data_path 'color_concat_430_650.png']);

%% make an sRGB image
fprintf('making the srgb image\n');
params.lambdas = lambdas;

% stuffed toy parameters
% wpatch: 0.6
% minv: 0.07
% maxv: 0.95

params.min_v = 0.02;%0.07;
params.max_v = 0.95; %1;%0.95;
im_srgb_pd = spec2srgb(msImg_pd_crop, params);
% im_srgb = spec2srgb(msImg_crop, params);

% imwrite(im_srgb_pd.^(2.2/1.5), 'Y:\papers\prism\scene\20170519_tulip2_with_gt\20170519_result_sh\5_refine\im_srgb_pd.png');
% im_srgb = spec2srgb(msImg, params);

figure; 
subplot(1,2,1); imshow(im_srgb_pd); title('pd');
% subplot(1,2,2); imshow(im_srgb); title('pref');


imwrite(im_srgb_pd, [data_path 'im_srgb_pd.png']);
% imwrite(im_srgb, [data_path 'im_srgb.png']);

%% do white balancing
% % figure;
% % title('select a white patch for WB');
% % [im_srgb_pd_srgb_white, im_srgb_pd_srgb_white_rect] = imcrop(im_srgb_pd);
% % white_vec = squeeze(mean(mean(im_srgb_pd_srgb_white,1),2));
% % params.sRratio = white_vec(1)/max(white_vec);
% % params.sGratio = white_vec(2)/max(white_vec);
% % params.sBratio = white_vec(3)/max(white_vec);
% params.min_v = 0; 0.03;
% params.max_v = 1;
% im_srgb_pd = spec2srgb(msImg_pd, params);
% im_srgb = spec2srgb(msImg, params);
% figure; 
% subplot(1,2,1); imshow(im_srgb_pd); title('pd with wb');
% subplot(1,2,2); imshow(im_srgb); title('pref with wb');
% 
% 
% 
% imwrite(im_srgb_pd, [data_path 'im_srgb_wb_pd.png']);
% imwrite(im_srgb, [data_path 'im_srgb_wb.png']);

%%
fprintf('Start the debugging process\n');
while true
    patch_est = zeros(nCh,1);
    

    figure(5000); imshow(im_srgb_pd);
    fprintf('Select a region of interest\n');
    [im_rect, rectobj] = imcrop;
    
    
    spec_plot_regions = round(rectobj);
    spec_plot_regions(:,[1,2]) = spec_plot_regions(:,[2,1]);
    spec_plot_regions(:,[2,3]) = spec_plot_regions(:,[3,2]);
    spec_plot_regions(:,[2,4]) = spec_plot_regions(:,[1,3]) + spec_plot_regions(:,[4,2]);
    
    % extract data
    figure(5001);
    patch_est(:) = squeeze( mean(mean( msImg_pd_crop(spec_plot_regions(1):spec_plot_regions(2), spec_plot_regions(3):spec_plot_regions(4), :))) );
    plot(lambdas, patch_est(:), '-.');
    axis([min(lambdas), max(lambdas), 0, 1]);
    xlabel('wavelength [nm]');
    ylabel('Normalized intensity');
    hold off;
    
    for j = 1:nCh
%         fprintf('%d\t %f\n', lambdas(j), patch_est(j));
        fprintf('%f\n', patch_est(j));
    end
    % global spec_plot_regions;
    % spec_plot_regions = rectarr;
end
