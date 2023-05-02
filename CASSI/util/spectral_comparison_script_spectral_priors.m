% Updated!
% PSNR, SSIM values with the corresponding hyperspectral data for a sample
% patch

close all;
clear;

data_path = 'Y:\papers\prism\scene\20170215_spectrum_recon_test\synthetic_vclab_colorchecker_downsampled2\comp_internal\';
gt_path = [data_path '..\msimgs\subset\'];
% gt_path = [data_path '..\GT\'];

files = dir(data_path);
dirFlags = [files.isdir];
folderlist = removerows(files(dirFlags), 'ind', [1,2]);

gt_srgb = imread([data_path 'gt_rgb.png']);

% num_roi = input('# of regions of interest: ');
lambdas = 430:10:650;

%% Read spectral image sets
fprintf('\tReading spectral image result sets\n');
num_dataset = numel(folderlist);
method_type = cell(numel(folderlist),1);

msImg = [];
gt_list = dir([gt_path '*.png']);
gt_list = gt_list(1:23);
% for i = 2:(numel(gt_list)-3)
for i = 1:numel(gt_list)
    msImg = cat(3, msImg, im2double(imread([gt_path gt_list(i).name])));
end
method_type{1} = 'GT';

for i = 1:num_dataset
    varname = [folderlist(i).name '_msimgs'];
    eval([varname ' = [];']);
    filelist = dir([data_path folderlist(i).name '\*.png']);
    nCh = numel(filelist);
    nCh = 23;
    for j = 1:nCh
        eval([varname ' = cat(3, ' varname ', im2double(imread([data_path folderlist(i).name ''\'' filelist(j).name ])));']);
%         eval([varname ' = ' varname '(1:23);']);
%         eval([folderlist(i).name '_psnr(' int2str(j) ') = psnr()im2double(imread([data_path folderlist(i).name ''\'' filelist(j).name ])));']);
    end
    method_type{i} = folderlist(i).name;
end

%% compute PSNR
PSNRs = zeros(num_dataset, nCh);
SSIMs = zeros(num_dataset, nCh);
for i = 1:num_dataset
    for j = 1:nCh
        eval(['PSNRs(i,j) = psnr(msImg(20:end-20 ,20:end-20,j), ' folderlist(i).name '_msimgs(20:end-20,20:end-20,j)' ');']);
        eval(['SSIMs(i,j) = ssim(msImg(20:end-20, 20:end-20,j), ' folderlist(i).name '_msimgs(20:end-20,20:end-20,j)' ');']);
    end
end
PSNR_methods = mean(PSNRs, 2);
SSIM_methods = mean(SSIMs, 2);

%%
patch_est_table = [];
rmse_table = [];
GT_table = [];
while true
    patch_est = zeros(nCh,num_dataset);
    patch_gt = zeros(nCh, 1);
    RMSE = zeros(num_dataset, 1);
    
    rectobj = zeros(1,4);
    figure(5000); imshow(gt_srgb);
    fprintf('Select a region of interest\n');
    [im_rect, rectobj] = imcrop;
    
    
    spec_plot_regions = round(rectobj);
    spec_plot_regions(:,[1,2]) = spec_plot_regions(:,[2,1]);
    spec_plot_regions(:,[2,3]) = spec_plot_regions(:,[3,2]);
    spec_plot_regions(:,[2,4]) = spec_plot_regions(:,[1,3]) + spec_plot_regions(:,[4,2]);
    
    % extract data
    figure;
    hold on;
    for j = 1:num_dataset
        %         rgb_vec_est = squeeze( mean(mean(im_spec_est(spec_plot_regions(i,1):spec_plot_regions(i,2), spec_plot_regions(i,3):spec_plot_regions(i,4), :))) );
        eval(['rgb_vec_est = squeeze( mean(mean(' [folderlist(j).name '_msimgs'] '(spec_plot_regions(1):spec_plot_regions(2), spec_plot_regions(3):spec_plot_regions(4), :))) );']);
        patch_est(:,j) = rgb_vec_est;
        plot(lambdas, rgb_vec_est, '-.');
    end
    rgb_vec_gt = squeeze( mean(mean(msImg(spec_plot_regions(1):spec_plot_regions(2), spec_plot_regions(3):spec_plot_regions(4), :))) );
    patch_gt = rgb_vec_gt;
    plot(lambdas, rgb_vec_gt, '-k.');
    legend(method_type);
    axis([min(lambdas), max(lambdas), 0, 1]);
    hold off;
    
    RMSE = zeros(1,num_dataset);
    for k = 1:num_dataset
        RMSE(k) = sqrt(mean((patch_est(:,k) - rgb_vec_gt).^2));
    end
    
    %%
    for k = 1:num_dataset
        fprintf('%s\n', method_type{k});
        for j = 1:nCh
            fprintf('%d\t %f\n', lambdas(j), patch_est(j,k));
        end
        fprintf('RMSE: %f\n\n', RMSE(k));
    end
    patch_est_table = cat(2,patch_est_table,patch_est);
    rmse_table = cat(2,rmse_table,RMSE);
    fprintf('GT\n');
    for j = 1:nCh
        fprintf('%d\t %f\n', lambdas(j), rgb_vec_gt(j));
    end
    GT_table = cat(2,GT_table,rgb_vec_gt);
    % global spec_plot_regions;
    % spec_plot_regions = rectarr;
end





%% Spectral plots
for i = 1:num_roi
%     subplot(R,C,i);
    figure;
    hold on;
    if ~isempty(msImg)
        rgb_vec_gt = squeeze( mean(mean(msImg(spec_plot_regions(i,1):spec_plot_regions(i,2), spec_plot_regions(i,3):spec_plot_regions(i,4), :))) );
        patch_gt(i,:) = rgb_vec_gt;
        plot(lambdas, rgb_vec_gt, '-k.');
        legend(method_type);
        for k = 1:num_dataset
            RMSE(i,k) = sqrt(mean((patch_est(i,:,k) - rgb_vec_gt').^2));
        end
        %         legend('EST', 'GT');
        %         title(sprintf('%d,%d, RMSE: %.2f', spec_plot_regions(i,1), spec_plot_regions(i,3), RMSE));
    else
        legend('EST');
    end
    hold off;
    xlabel('wavelength [nm]');
    ylabel('intensity');
    axis([min(lambdas),max(lambdas),0,1]);
end
drawnow;