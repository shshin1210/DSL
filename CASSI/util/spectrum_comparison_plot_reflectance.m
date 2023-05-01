function [fh, patch_est, patch_gt] = spectrum_comparison_plot_reflectance( im_spec_est, msImg, lambdas)
global ref_white_region;
if isempty(ref_white_region)
    figure; imshow(im_spec_est(:,:,round(end/2)));
    rect = getrect(gcf);
    min_r = round(rect(2));
    max_r = round(rect(2) + rect(4) - 1);
    min_c = round(rect(1));
    max_c = round(rect(1) + rect(3) - 1);
    
    ref_white_region = [min_r, max_r, min_c, max_c];
end

min_r = ref_white_region(1);
max_r = ref_white_region(2);
min_c = ref_white_region(3);
max_c = ref_white_region(4);
ref_spec_gt = (mean(mean(msImg(min_r:max_r, min_c:max_c, :),1),2))*1.1;
ref_spec_est = (mean(mean(im_spec_est(min_r:max_r, min_c:max_c, :),1),2))*1.1;

im_spec_est = bsxfun(@rdivide, im_spec_est, ref_spec_est);
msImg = bsxfun(@rdivide, msImg, ref_spec_gt);

global valid_lambdas_ind;
if isempty(valid_lambdas_ind)
    valid_lambdas_ind = 1:size(im_spec_est,3);
end

im_spec_est = im_spec_est(:,:,valid_lambdas_ind);
msImg = msImg(:,:,valid_lambdas_ind);
lambdas = lambdas(valid_lambdas_ind);

im_spec_est = (im_spec_est /mean(im_spec_est(:)))*mean(msImg(:));
im_spec_est = max(min(im_spec_est,1),0);
%SPECTRUM_COMPARISON_PLOT Plot a colorchart in multiple multispectral images
%   Plot a colorchart in multiple multispectral images.
% 

% num_varargin = length(varargin);

% c_min = 130; c_max = 154;
% r_min = 319; r_max = 343;
% r_step = 46; c_step = 46;

% c_min = 129; c_max = 144;
% r_min = 76; r_max = 93;
% r_step = 39; c_step = 43;

c_min = 135; c_max = 158;
r_min = 110; r_max = 145;
r_step = 85; c_step = 85;

% c_min = coordList.c_min;    c_max = coordList.c_max;
% r_min = coordList.r_min;    r_max = coordList.r_max;
% r_step = coordList.r_step;    c_step = coordList.c_step;
% display_spectral_images_graph(im_spec_smit(r_min:r_max, c_min:c_max,:));

[~,~,nCh] = size(im_spec_est);

% visualize the target region of color patches
debug_img = repmat(msImg(:,:,1),1,1,3);
for r = 1:4
    for c = 1:6
        debug_img((r_min:r_max) + (r-1)*r_step, c_min + (c-1)*c_step, 1) = 1;
        debug_img((r_min:r_max) + (r-1)*r_step, c_min + (c-1)*c_step, 2) = 0;
        debug_img((r_min:r_max) + (r-1)*r_step, c_min + (c-1)*c_step, 3) = 0;
        
        debug_img((r_min:r_max) + (r-1)*r_step, c_max + (c-1)*c_step, 1) = 1;
        debug_img((r_min:r_max) + (r-1)*r_step, c_max + (c-1)*c_step, 2) = 0;
        debug_img((r_min:r_max) + (r-1)*r_step, c_max + (c-1)*c_step, 3) = 0;
        
        debug_img(r_min + (r-1)*r_step, (c_min:c_max) + (c-1)*c_step, 1) = 1;
        debug_img(r_min + (r-1)*r_step, (c_min:c_max) + (c-1)*c_step, 2) = 0;
        debug_img(r_min + (r-1)*r_step, (c_min:c_max) + (c-1)*c_step, 3) = 0;
        
        debug_img(r_max + (r-1)*r_step, (c_min:c_max) + (c-1)*c_step, 1) = 1;
        debug_img(r_max + (r-1)*r_step, (c_min:c_max) + (c-1)*c_step, 2) = 0;
        debug_img(r_max + (r-1)*r_step, (c_min:c_max) + (c-1)*c_step, 3) = 0;
    end
end
figure; imshow(debug_img);
title(int2str(lambdas(1)));



%fh = figure(300);
fh = figure;
set(fh, 'Position', [100, 100, 1500, 1200]);
ind = 1;
% RMSE_val = zeros(4,6);
patch_est = zeros(4,6,nCh);
patch_gt = zeros(4,6,nCh);

for r = 1:4
    for c = 1:6
        % for each color patch
        %
%         if ind > nCh
%             break;
%         end
        rgb_vec_est = squeeze( mean(mean(im_spec_est((r_min:r_max) + (r-1)*r_step, (c_min:c_max) + (c-1)*c_step, :))) );
        patch_est(r,c,:) = rgb_vec_est;
%         rgb_vec_tM = squeeze( mean(mean(im_spec_tM((r_min:r_max) + (r-1)*r_step, (c_min:c_max) + (c-1)*c_step, :))) );
        rgb_vec_gt = squeeze( mean(mean(msImg((r_min:r_max) + (r-1)*r_step, (c_min:c_max) + (c-1)*c_step, :))) );
        patch_gt(r,c,:) = rgb_vec_gt;
        subplot(4,6,ind);
        hold on;
        plot(lambdas, rgb_vec_est, '--r.');
        %         plot(rgb_vec_tM, '--m*');
        plot(lambdas, rgb_vec_gt, '--k.');
        hold off;
        %         legend('SM', 'TR', 'GT');
        legend('EST', 'GT');
        RMSE = sqrt(mean((rgb_vec_est - rgb_vec_gt).^2));
        title(sprintf('RMSE: %.4f', RMSE));
        xlabel('wavelength [nm]');
        ylabel('reflectance');
        axis([min(lambdas),max(lambdas),0,1]);
        
        ind = ind + 1;
    end
%     if ind > nCh
%         break;
%     end
end
drawnow;

end

