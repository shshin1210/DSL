close all;
clear;

% data_path = 'Y:\papers\prism\scene\20170508_indoor_plants3\result_ic1\5_refine\';
data_path = 'C:\Users\S.H. BAEK\Desktop\tmp\';
camera_path = 'Y:\papers\prism\calib\20170429_calib\camera\';
load([camera_path 'cam_calib.mat']);
model_path = 'Y:\papers\prism\calib\20170429_calib\model\';
unwarp_model_fn = 'p_ref.mat';

% load camera parameters
folder_name = uigetdir;
folder_name = [folder_name '\'];

fn_list = dir([folder_name '*0.png']);
fn_list = fn_list(1:23);

load([model_path unwarp_model_fn], 'p_ref');

[n_pd_y, n_pd_x, ~] = size(p_ref);
p_ref(:,:,1) = round(p_ref(:,:,1) + cam.center_col);
p_ref(:,:,2) = round(p_ref(:,:,2) + cam.center_row);
figure; subplot(1,2,1); imagesc(p_ref(:,:,1)); colorbar; subplot(1,2,2); imagesc(p_ref(:,:,2)); colorbar;

% apply cropping and downsampling
downsampling_ratio = 4;
min_r = 490; max_r = 3300; min_c = 552; max_c = 5790;
p_ref(:,:,1) = min(p_ref(:,:,1) - min_c, max_c-min_c+1) / downsampling_ratio;
p_ref(:,:,2) = min(p_ref(:,:,2) - min_r, max_r-min_r+1) / downsampling_ratio;
figure; subplot(1,2,1); imagesc(p_ref(:,:,1)); colorbar; subplot(1,2,2); imagesc(p_ref(:,:,2)); colorbar;


msImg_pd = convert_pref_to_pd(msImg, p_ref);