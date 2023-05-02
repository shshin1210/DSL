% USE verseion 2 instead of this file

close all;
clear;

data_path = 'Y:\papers\prism\scene\20170215_spectrum_recon_test\synthetic_columbia_stuffed_toys_with_real_spec2rgb\comp_system_fig\';
gt_path = [data_path '..\msimgs\'];

files = dir(data_path);
dirFlags = [files.isdir];
folderlist = removerows(files(dirFlags), 'ind', [1,2]);

gt_srgb = imread([data_path 'gt_srgb.png']);

num_roi = input('# of regions of interest: ');
rectobj = zeros(num_roi,4);
figure; imshow(gt_srgb);
fprintf('Select a region of interest\n');
lambdas = 430:10:650;

for i = 1:num_roi        
    [~, rectobj(i,:)] = imcrop;
end

close;
spec_plot_regions = round(rectobj);
spec_plot_regions(:,[1,2]) = spec_plot_regions(:,[2,1]);
spec_plot_regions(:,[2,3]) = spec_plot_regions(:,[3,2]);
spec_plot_regions(:,[2,4]) = spec_plot_regions(:,[1,3]) + spec_plot_regions(:,[4,2]);
% global spec_plot_regions;
% spec_plot_regions = rectarr;

%% Read spectral image sets
fprintf('\tReading spectral image result sets\n');
num_dataset = numel(folderlist);
method_type = cell(numel(folderlist),1);
for i = 1:num_dataset
    varname = [folderlist(i).name '_msimgs'];
    eval([varname ' = [];']);
    filelist = dir([data_path folderlist(i).name '\*0.png']);
    nCh = numel(filelist);
    for j = 1:nCh
        eval([varname ' = cat(3, ' varname ', im2double(imread([data_path folderlist(i).name ''\'' filelist(j).name ])));']);
    end
    method_type{i} = folderlist(i).name;
end

msImg = [];
gt_list = dir([gt_path '*.png']);
for i = 2:(numel(gt_list)-3)
    msImg = cat(3, msImg, im2double(imread([gt_path gt_list(i).name])));
end
method_type{end+1} = 'GT';

fh = figure;
set(fh, 'Position', [100, 100, 1500, 1200]);
patch_est = zeros(num_roi,nCh,num_dataset);
patch_gt = zeros(num_roi,nCh);
RMSE = zeros(num_roi, num_dataset);

% [R, C] = findIntegerFactorsCloseToSquarRoot(num_roi);

%% Spectral plots
for i = 1:num_roi
%     subplot(R,C,i);
    figure;
    hold on;
    for j = 1:num_dataset
%         rgb_vec_est = squeeze( mean(mean(im_spec_est(spec_plot_regions(i,1):spec_plot_regions(i,2), spec_plot_regions(i,3):spec_plot_regions(i,4), :))) );
        eval(['rgb_vec_est = squeeze( mean(mean(' [folderlist(j).name '_msimgs'] '(spec_plot_regions(i,1):spec_plot_regions(i,2), spec_plot_regions(i,3):spec_plot_regions(i,4), :))) );']);
        patch_est(i,:,j) = rgb_vec_est;
        plot(lambdas, rgb_vec_est, '-.');
    end
    
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