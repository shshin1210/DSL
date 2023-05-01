function [fh, patch_est, patch_gt] = spectrum_comparison_plot_given_region( im_spec_est, msImg, lambdas )
%SPECTRUM_COMPARISON_PLOT Plot a colorchart in multiple multispectral images
%   Plot a colorchart in multiple multispectral images.
% region: [ min_r1, max_r1, min_c1, max_c1 ]
%       : [ min_r2, max_r2, min_c2, max_c2 ]
%       :         ...

% num_varargin = length(varargin);

global spec_plot_regions;
addpath('..\util\');

if isempty(spec_plot_regions)
    params_.lambdas = lambdas;
    vis_img = spec2srgb(im_spec_est, params_);
    figure; imshow(vis_img);
    
    nRegions = input('enter the number of samples for debugging: ');
    spec_plot_regions = zeros(nRegions, 4);
    for i = 1:nRegions
        hold on;
        [~, rect] = imcrop(vis_img);
        rect = round(rect);
        hold off;
        spec_plot_regions(i,:) = [rect(2), rect(2) + rect(4) - 1, rect(1), rect(1) + rect(3) - 1];
    end
    close(gcf);
end


nRegions = size(spec_plot_regions, 1);

[~,~,nCh] = size(im_spec_est);

% visualize the target region of color patches
debug_img = repmat(im_spec_est(:,:,1),1,1,3);
for i = 1:nRegions
    debug_img(spec_plot_regions(i,1):spec_plot_regions(i,2), spec_plot_regions(i,3), 1) = 1;
    debug_img(spec_plot_regions(i,1):spec_plot_regions(i,2), spec_plot_regions(i,3), 2) = 0;
    debug_img(spec_plot_regions(i,1):spec_plot_regions(i,2), spec_plot_regions(i,3), 3) = 0;
    
    debug_img(spec_plot_regions(i,1):spec_plot_regions(i,2), spec_plot_regions(i,4), 1) = 1;
    debug_img(spec_plot_regions(i,1):spec_plot_regions(i,2), spec_plot_regions(i,4), 2) = 0;
    debug_img(spec_plot_regions(i,1):spec_plot_regions(i,2), spec_plot_regions(i,4), 3) = 0;
    
    debug_img(spec_plot_regions(i,1), spec_plot_regions(i,3):spec_plot_regions(i,4), 1) = 1;
    debug_img(spec_plot_regions(i,1), spec_plot_regions(i,3):spec_plot_regions(i,4), 2) = 0;
    debug_img(spec_plot_regions(i,1), spec_plot_regions(i,3):spec_plot_regions(i,4), 3) = 0;
    
    debug_img(spec_plot_regions(i,2), spec_plot_regions(i,3):spec_plot_regions(i,4), 1) = 1;
    debug_img(spec_plot_regions(i,2), spec_plot_regions(i,3):spec_plot_regions(i,4), 2) = 0;
    debug_img(spec_plot_regions(i,2), spec_plot_regions(i,3):spec_plot_regions(i,4), 3) = 0;
end
% figure; imshow(debug_img);
% title(int2str(lambdas(1)));
% 
fh = figure;
set(fh, 'Position', [100, 100, 1500, 1200]);
patch_est = zeros(nRegions,nCh);
patch_gt = zeros(nRegions,nCh);

% R = ceil(sqrt(nRegions));
% C = R;
% R = 4;  C = 6;
[R, C] =  findIntegerFactorsCloseToSquarRoot(nRegions);

for i = 1:nRegions
    rgb_vec_est = squeeze( mean(mean(im_spec_est(spec_plot_regions(i,1):spec_plot_regions(i,2), spec_plot_regions(i,3):spec_plot_regions(i,4), :))) );
    patch_est(i,:) = rgb_vec_est;

    subplot(R,C,i);
    hold on;
    plot(lambdas, rgb_vec_est, '-r.');
    if ~isempty(msImg)
        rgb_vec_gt = squeeze( mean(mean(msImg(spec_plot_regions(i,1):spec_plot_regions(i,2), spec_plot_regions(i,3):spec_plot_regions(i,4), :))) );
        patch_gt(i,:) = rgb_vec_gt;
        plot(lambdas, rgb_vec_gt, '-k.');
        legend('EST', 'GT');
        RMSE = sqrt(mean((rgb_vec_est - rgb_vec_gt).^2));
        title(sprintf('%d,%d, RMSE: %.2f', spec_plot_regions(i,1), spec_plot_regions(i,3), RMSE));
    else
        legend('EST');
    end
    hold off;
    xlabel('wavelength [nm]');
    ylabel('intensity');
    axis([min(lambdas),max(lambdas),0,1]);
end
drawnow;

end

function [a, b] =  findIntegerFactorsCloseToSquarRoot(n)
% a cannot be greater than the square root of n
% b cannot be smaller than the square root of n
% we get the maximum allowed value of a
amax = floor(sqrt(n));
if 0 == rem(n, amax)
    % special case where n is a square number
    a = amax;
    b = n / a;
    return;
end
% Get its prime factors of n
primeFactors  = factor(n);
% Start with a factor 1 in the list of candidates for a
candidates = [1];
for i=1:numel(primeFactors)
    % get the next prime factr
    f = primeFactors(i);
    % Add new candidates which are obtained by multiplying
    % existing candidates with the new prime factor f
    % Set union ensures that duplicate candidates are removed
    candidates  = union(candidates, f .* candidates);
    % throw out candidates which are larger than amax
    candidates(candidates > amax) = [];
end
% Take the largest factor in the list d
a = candidates(end);
b = n / a;
end

